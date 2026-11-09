"""Expérience « deep learning » : réseau de neurones face au modèle actuel (phase 4).

Protocole identique à l'entraînement de production (mêmes plis walk-forward, même
calibration par course, mêmes courses comparées) ; rien n'est enregistré en base.

Critère de succès fixé AVANT de voir les résultats : `neural_market` (ou l'empilement
`stack_market`) doit battre `lgbm_market` en log-loss avec un IC 95 % entièrement
négatif. Sinon, le modèle actuel est conservé.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from horse_intelligence.evaluation.metrics import compute_metrics, paired_bootstrap
from horse_intelligence.evaluation.walk_forward import ModelRunner, run_walk_forward
from horse_intelligence.intelligence.feature_engineering import (
    CATEGORICAL,
    build_dataset,
    load_raw,
)
from horse_intelligence.intelligence.leakage_audit import audit_source_aggregates
from horse_intelligence.models.lightgbm_model import LgbmRaceModel
from horse_intelligence.models.neural_model import NeuralRaceModel
from horse_intelligence.models.ranking import ConditionalLogit, FloatArray
from horse_intelligence.services.training import (
    LGBM_FEATURES,
    NUMERIC_FEATURES,
    REFERENCE_MODEL,
    _masked,
    _with_market,
    model_runners,
)

logger = logging.getLogger(__name__)

CHALLENGERS = ("neural_market", "stack_market")
INCUMBENT = "lgbm_market"


@dataclass
class ExperimentReport:
    period: tuple[str, str]
    n_races: int
    n_folds: int
    seconds: float
    metrics: dict[str, dict[str, float]]
    vs_reference: dict[str, dict[str, float]]
    vs_incumbent: dict[str, dict[str, float]]
    nn_fits: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, list[str]] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return any(self.vs_incumbent.get(c, {}).get("ci_high", 0.0) < 0 for c in CHALLENGERS)


def neural_runners(refit_every_days: int = 7,
                   fits: list[dict[str, Any]] | None = None) -> dict[str, ModelRunner]:
    """Le réseau (coûteux) n'est réappris que tous les `refit_every_days` jours de données.

    Sans fuite : un réseau appris sur l'apprentissage d'un pli ne voit que des dates
    antérieures à la calibration et au test de ce pli comme des plis suivants. La
    calibration, elle, est refaite à chaque pli.
    """
    state: dict[str, Any] = {"model": None, "until": None}
    log = fits if fits is not None else []

    def score_model(base: pd.DataFrame) -> NeuralRaceModel:
        last: date = base["race_date"].max()
        if state["model"] is None or (last - state["until"]).days >= refit_every_days:
            t0 = time.perf_counter()
            state["model"] = NeuralRaceModel(NUMERIC_FEATURES, CATEGORICAL).fit_score(base)
            state["until"] = last
            log.append({"trained_until": str(last), "epochs": state["model"].epochs_run,
                        "seconds": round(time.perf_counter() - t0, 1)})
        model: NeuralRaceModel = state["model"]
        return model

    def neural_market(base: pd.DataFrame, calib: pd.DataFrame,
                      test: pd.DataFrame) -> FloatArray:
        m = replace(score_model(base), with_market=True, calibrator=None)
        m.fit_calibration(_with_market(calib))
        sub = _with_market(test)
        return _masked(test, sub, m.predict(sub))

    def stack_market(base: pd.DataFrame, calib: pd.DataFrame,
                     test: pd.DataFrame) -> FloatArray:
        nn = score_model(base)
        cal, sub = _with_market(calib), _with_market(test)
        lgbm = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL, with_market=True).fit(base, cal)

        def inputs(f: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"lgbm_score": lgbm.raw_score(f), "nn_score": nn.raw_score(f),
                                 "mkt_log_prob": f["mkt_log_prob"].to_numpy()}, index=f.index)

        z = inputs(cal)
        stack = ConditionalLogit(l2=1e-3).fit(z, cal["race_id"], cal["won"], list(z.columns))
        return _masked(test, sub, stack.predict_proba(inputs(sub), sub["race_id"]))

    return {"neural_market": neural_market, "stack_market": stack_market}


def run_neural_experiment(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                          calib_days: int = 7, refit_every_days: int = 7) -> ExperimentReport:
    t0 = time.perf_counter()
    raw = load_raw(engine)
    ds = build_dataset(raw)
    if not audit_source_aggregates(ds.frame).passed:
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]
    base_runners = model_runners()
    fits: list[dict[str, Any]] = []
    runners = {"market": base_runners["market"], REFERENCE_MODEL: base_runners[REFERENCE_MODEL],
               INCUMBENT: base_runners[INCUMBENT],
               **neural_runners(refit_every_days, fits)}
    wf = run_walk_forward(frame, runners, min_train_days, test_days, calib_days)
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = list(runners)
    mkt = _with_market(evald)
    mkt = mkt[mkt.groupby("race_id")[names].transform(lambda s: s.notna().all()).all(axis=1)]
    dates = sorted(frame["race_date"].unique())
    return ExperimentReport(
        period=(str(dates[0]), str(dates[-1])), n_races=int(mkt["race_id"].nunique()),
        n_folds=len(wf.folds), seconds=round(time.perf_counter() - t0, 1),
        metrics={n: compute_metrics(mkt, mkt[n]).as_dict() for n in names},
        vs_reference={n: paired_bootstrap(mkt, mkt[n], mkt[REFERENCE_MODEL])
                      for n in names if n != REFERENCE_MODEL},
        vs_incumbent={n: paired_bootstrap(mkt, mkt[n], mkt[INCUMBENT]) for n in CHALLENGERS},
        nn_fits=fits, errors=wf.errors)


def run_combo_backtest(engine: Engine, model_col: str = INCUMBENT,
                       market_col: str = REFERENCE_MODEL) -> dict[str, Any]:
    """Paris combinés simulés sur les prédictions hors échantillon du dernier entraînement."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.combo_simulator import (
        COMBO_BETS,
        build_candidates,
        reference_strategies,
        select_then_test_combos,
        simulate_combos,
    )
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez scripts/train_models.py")
    pred = _with_market(pred)
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    cand = build_candidates(pred, payouts, model_col, market_col).table
    out: dict[str, Any] = {"period": [str(pred["race_date"].min()), str(pred["race_date"].max())],
                           "bets": {}}
    for bt in COMBO_BETS:
        if cand.empty or not (cand["bet_type"] == bt).any():
            continue
        whole = cand[cand["bet_type"] == bt]
        out["bets"][bt] = {
            "all_period": [asdict(simulate_combos(whole, s)) for s in reference_strategies(bt)],
            "selection": asdict(select_then_test_combos(cand, bt)),
        }
    return out


def run_rank_strategies(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Simple gagnant/placé sur le k-ième cheval du modèle ou du marché, toutes courses
    ou quinté seulement, sur les prédictions hors échantillon du dernier entraînement."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.rank_strategies import (
        rank_strategy_table,
        robust_winners,
    )
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez scripts/train_models.py")
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    rows = rank_strategy_table(_with_market(pred), payouts, model_col)
    return {"period": [str(pred["race_date"].min()), str(pred["race_date"].max())],
            "rules": rows, "robust_winners": robust_winners(rows)}
