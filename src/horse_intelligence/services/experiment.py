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

import numpy as np
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


def run_rank_strategies(engine: Engine, model_col: str = INCUMBENT, start: date | None = None,
                        end: date | None = None) -> dict[str, Any]:
    """Simple gagnant/placé sur le k-ième cheval du modèle ou du marché (toutes courses,
    quinté ou pick 5), sur les prédictions hors échantillon du dernier entraînement,
    éventuellement limitées à [start, end] (validation d'une règle figée sur une période
    qu'elle n'a pas vue)."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.rank_strategies import (
        rank_strategy_table,
        robust_winners,
    )
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez scripts/train_models.py")
    if start is not None:
        pred = pred[pred["race_date"] >= start]
    if end is not None:
        pred = pred[pred["race_date"] <= end]
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    rows = rank_strategy_table(_with_market(pred), payouts, model_col)
    return {"period": [str(pred["race_date"].min()), str(pred["race_date"].max())],
            "rules": rows, "robust_winners": robust_winners(rows)}


def run_segment_analysis(engine: Engine, model_col: str = INCUMBENT,
                         market_col: str = REFERENCE_MODEL) -> dict[str, pd.DataFrame]:
    """Analyse par discipline, partants, distance, réunion, groupe et rang du modèle."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.segment_analysis import (
        model_value_by_segment,
        segment_table,
    )
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez scripts/train_models.py")
    pred = _with_market(pred)
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
        races = pd.read_sql(text("SELECT race_id, reunion_num, label, conditions_text "
                                 "FROM races"), conn)
    frame = pred.merge(races, on="race_id", how="left")
    return {"roi": segment_table(frame, payouts, model_col),
            "valeur_modele": model_value_by_segment(frame, model_col, market_col),
            "periode": pd.DataFrame({"debut": [str(pred["race_date"].min())],
                                     "fin": [str(pred["race_date"].max())]})}


def run_frozen_rules(engine: Engine, model_col: str = INCUMBENT, start: date | None = None,
                     end: date | None = None) -> dict[str, Any]:
    """Confronte les règles figées le 10/11/2026 à la période [start, end]."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.frozen_rules import evaluate_frozen_rules
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez scripts/train_models.py")
    pred = _with_market(pred)
    if start is not None:
        pred = pred[pred["race_date"] >= start]
    if end is not None:
        pred = pred[pred["race_date"] <= end]
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
        races = pd.read_sql(text("SELECT race_id, reunion_num, label, conditions_text "
                                 "FROM races"), conn)
    frame = pred.merge(races, on="race_id", how="left")
    return {"period": [str(pred["race_date"].min()), str(pred["race_date"].max())],
            "rules": evaluate_frozen_rules(frame, payouts, model_col)}


def run_bankroll(engine: Engine, model_col: str = INCUMBENT, rule_value: str = "Trot G",
                 n_sims: int = 10_000) -> dict[str, Any]:
    """Simulation de bankroll de la règle figée « 3e du modèle, gagnant, trot G » (ou d'une
    autre règle figée désignée par sa valeur de segment) à partir de ses paris historiques
    hors échantillon, réglés aux rapports officiels."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.bankroll import run_scenarios
    from horse_intelligence.evaluation.frozen_rules import FROZEN_RULES, rule_bets
    from horse_intelligence.services.training import load_oos_predictions

    rule = next(r for r in FROZEN_RULES if r.value == rule_value)
    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez scripts/train_models.py")
    pred = _with_market(pred)
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
        races = pd.read_sql(text("SELECT race_id, reunion_num, label, conditions_text "
                                 "FROM races"), conn)
    bets = rule_bets(pred.merge(races, on="race_id", how="left"), payouts, model_col, rule)
    if bets.empty:
        raise ValueError("aucun pari historique pour cette règle")
    first, last = bets["race_date"].min(), bets["race_date"].max()
    years = max((last - first).days, 1) / 365.25
    per_year = round(len(bets) / years)
    profits = bets["profit"].to_numpy()
    # parcours réel, dans l'ordre chronologique
    cum = profits.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    losing = profits <= -1.0 + 1e-12
    streak = best = 0
    for lost in losing:
        streak = streak + 1 if lost else 0
        best = max(best, streak)
    by_year = bets.assign(year=pd.to_datetime(bets["race_date"]).dt.year).groupby("year")[
        "profit"].agg(["count", "sum", "mean"])
    return {
        "rule": rule.name, "period": [str(first), str(last)], "n_bets": len(bets),
        "bets_per_year": per_year, "hit_rate": float((profits > -1).mean()),
        "roi": float(profits.mean()),
        "best_payout": float(profits.max() + 1.0),
        "median_winning_payout": float(np.median(profits[profits > -1] + 1.0))
        if (profits > -1).any() else float("nan"),
        "real_path": {"final": float(cum[-1]), "max_drawdown": float((peak - cum).max()),
                      "longest_losing_streak": int(best), "lowest": float(min(cum.min(), 0.0))},
        "real_by_year": {int(str(y)): {"paris": int(r["count"]), "gain": float(r["sum"]),
                                       "roi": float(r["mean"])}
                         for y, r in by_year.iterrows()},
        "scenarios": run_scenarios(profits, per_year, n_sims),
    }


FEATURES_PAIRS = (("lgbm_market", "lgbm_market_v2"), ("lgbm_form", "lgbm_form_v2"))


def features_runners(importances: dict[str, pd.Series]) -> dict[str, ModelRunner]:
    """Modèles avec les variables v2 (vitesse, ferrure, aptitudes, forme récente) en plus.
    Même protocole que la production ; l'importance des variables du dernier pli est gardée."""
    from horse_intelligence.intelligence.feature_engineering import V2_FEATURES

    feats = [*LGBM_FEATURES, *V2_FEATURES]

    def lgbm_market_v2(base: pd.DataFrame, calib: pd.DataFrame,
                       test: pd.DataFrame) -> FloatArray:
        m = LgbmRaceModel(feats, CATEGORICAL, with_market=True)
        m.fit(base, _with_market(calib))
        importances["lgbm_market_v2"] = m.feature_importance()
        sub = _with_market(test)
        return _masked(test, sub, m.predict(sub))

    def lgbm_form_v2(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        m = LgbmRaceModel(feats, CATEGORICAL).fit(base, calib)
        importances["lgbm_form_v2"] = m.feature_importance()
        return m.predict(test)

    return {"lgbm_market_v2": lgbm_market_v2, "lgbm_form_v2": lgbm_form_v2}


def run_features_experiment(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                            calib_days: int = 7) -> dict[str, Any]:
    """Les variables v2 améliorent-elles la prévision ? Critère fixé AVANT de voir les
    résultats : `lgbm_market_v2` doit battre `lgbm_market` en log-loss, IC 95 % < 0.
    Puis, sur les prédictions hors échantillon : sélection / test des règles de mise
    (mêmes grilles que la validation) et stratégies par rang."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.betting_simulator import select_then_test
    from horse_intelligence.evaluation.rank_strategies import (
        rank_strategy_table,
        robust_winners,
    )

    t0 = time.perf_counter()
    raw = load_raw(engine)
    ds = build_dataset(raw)
    if not audit_source_aggregates(ds.frame).passed:
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]
    base_runners = model_runners()
    importances: dict[str, pd.Series] = {}
    runners = {REFERENCE_MODEL: base_runners[REFERENCE_MODEL],
               "lgbm_market": base_runners["lgbm_market"],
               "lgbm_form": base_runners["lgbm_form"], **features_runners(importances)}
    wf = run_walk_forward(frame, runners, min_train_days, test_days, calib_days)
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = list(runners)
    mkt = _with_market(evald)
    mkt = mkt[mkt.groupby("race_id")[names].transform(lambda s: s.notna().all()).all(axis=1)]
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    paired = {new: paired_bootstrap(mkt, mkt[new], mkt[old]) for old, new in FEATURES_PAIRS}
    rank_rows = rank_strategy_table(mkt, payouts, "lgbm_market_v2")
    dates = sorted(frame["race_date"].unique())
    return {
        "period": [str(dates[0]), str(dates[-1])], "n_races": int(mkt["race_id"].nunique()),
        "n_folds": len(wf.folds), "seconds": round(time.perf_counter() - t0, 1),
        "metrics": {n: compute_metrics(mkt, mkt[n]).as_dict() for n in names},
        "vs_reference": {n: paired_bootstrap(mkt, mkt[n], mkt[REFERENCE_MODEL])
                         for n in names if n != REFERENCE_MODEL},
        "v2_vs_v1": paired,
        "success": paired["lgbm_market_v2"]["ci_high"] < 0,
        # le modèle « marché » réutilise le même score LightGBM que le modèle « forme »
        # (puis le combine aux cotes) : une seule liste d'importances suffit
        "importance": {k: v.sort_values(ascending=False).head(25).round(1).to_dict()
                       for k, v in importances.items() if k == "lgbm_form_v2"},
        "betting": {m: asdict(select_then_test(mkt, payouts, m))
                    for m in ("lgbm_market", "lgbm_market_v2")},
        "ranks_robust": robust_winners(rank_rows),
        "ranks_best_test": sorted([r for r in rank_rows if r["test"]["n_bets"] > 0],
                                  key=lambda r: -r["test"]["roi"])[:8],
        "errors": wf.errors,
    }
