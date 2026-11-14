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
    LGBM_FEATURES_V1,
    LGBM_FEATURES_V2,
    NUMERIC_FEATURES_V1,
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
            state["model"] = NeuralRaceModel(NUMERIC_FEATURES_V1, CATEGORICAL).fit_score(base)
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
        lgbm = LgbmRaceModel(LGBM_FEATURES_V1, CATEGORICAL, with_market=True).fit(base, cal)

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


def features_runners(importances: dict[str, pd.Series], old_feats: list[str] | None = None,
                     new_feats: list[str] | None = None,
                     tag: str = "v2") -> dict[str, ModelRunner]:
    """Modèles avec l'ancien jeu de variables (`lgbm_market`, `lgbm_form`) et le nouveau
    (`lgbm_market_<tag>`, `lgbm_form_<tag>`). Même protocole que la production ;
    importance des variables du dernier pli. Par défaut : v1 contre v1 + v2."""
    old = old_feats if old_feats is not None else LGBM_FEATURES_V1
    new = new_feats if new_feats is not None else LGBM_FEATURES_V2

    def market(feats: list[str]) -> ModelRunner:
        def run(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
            m = LgbmRaceModel(feats, CATEGORICAL, with_market=True)
            m.fit(base, _with_market(calib))
            sub = _with_market(test)
            return _masked(test, sub, m.predict(sub))
        return run

    def form(feats: list[str], key: str | None) -> ModelRunner:
        def run(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
            m = LgbmRaceModel(feats, CATEGORICAL).fit(base, calib)
            if key is not None:
                importances[key] = m.feature_importance()
            return m.predict(test)
        return run

    return {"lgbm_market": market(old), "lgbm_form": form(old, None),
            f"lgbm_market_{tag}": market(new),
            f"lgbm_form_{tag}": form(new, f"lgbm_form_{tag}")}


def run_features_experiment(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                            calib_days: int = 7, stage: str = "v2") -> dict[str, Any]:
    """Un nouveau jeu de variables améliore-t-il la prévision ? `stage="v2"` : variables
    d'origine contre v1 + v2 ; `stage="v3"` : production (v1 + v2) contre + v3. Critère fixé
    AVANT de voir les résultats : `lgbm_market_<stage>` bat `lgbm_market`, IC 95 % < 0.
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
    # v2 : variables d'origine contre v1 + v2 ; v3 : v1 + v2 contre v1 + v2 + v3
    old_feats, new_feats = ((LGBM_FEATURES_V1, LGBM_FEATURES_V2) if stage == "v2" else
                            (LGBM_FEATURES_V2, LGBM_FEATURES))
    new_m, new_f = f"lgbm_market_{stage}", f"lgbm_form_{stage}"
    pairs = (("lgbm_market", new_m), ("lgbm_form", new_f))
    runners = {REFERENCE_MODEL: base_runners[REFERENCE_MODEL],
               **features_runners(importances, old_feats, new_feats, stage)}
    wf = run_walk_forward(frame, runners, min_train_days, test_days, calib_days)
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = list(runners)
    mkt = _with_market(evald)
    mkt = mkt[mkt.groupby("race_id")[names].transform(lambda s: s.notna().all()).all(axis=1)]
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    paired = {new: paired_bootstrap(mkt, mkt[new], mkt[old]) for old, new in pairs}
    rank_rows = rank_strategy_table(mkt, payouts, new_m)
    with engine.connect() as conn:
        races = pd.read_sql(text("SELECT race_id, label, conditions_text FROM races"), conn)
    frozen = frozen_rule_by_period(mkt.merge(races, on="race_id", how="left"), payouts,
                                   ("lgbm_market", new_m))
    dates = sorted(frame["race_date"].unique())
    return {
        "period": [str(dates[0]), str(dates[-1])], "n_races": int(mkt["race_id"].nunique()),
        "n_folds": len(wf.folds), "seconds": round(time.perf_counter() - t0, 1),
        "metrics": {n: compute_metrics(mkt, mkt[n]).as_dict() for n in names},
        "vs_reference": {n: paired_bootstrap(mkt, mkt[n], mkt[REFERENCE_MODEL])
                         for n in names if n != REFERENCE_MODEL},
        "stage": stage, "challenger": new_m,
        "new_vs_old": paired,
        "success": paired[new_m]["ci_high"] < 0,
        # le modèle « marché » réutilise le même score LightGBM que le modèle « forme »
        # (puis le combine aux cotes) : une seule liste d'importances suffit
        "importance": {k: v.sort_values(ascending=False).head(25).round(1).to_dict()
                       for k, v in importances.items() if k == new_f},
        "betting": {m: asdict(select_then_test(mkt, payouts, m))
                    for m in ("lgbm_market", new_m)},
        "ranks_robust": robust_winners(rank_rows),
        "frozen_trot_g": frozen,
        "ranks_best_test": sorted([r for r in rank_rows if r["test"]["n_bets"] > 0],
                                  key=lambda r: -r["test"]["roi"])[:8],
        "errors": wf.errors,
    }


FROZEN_SPLIT = "2025-11-19"  # fin de l'année de validation des règles figées le 10/11/2026


def frozen_rule_by_period(frame: pd.DataFrame, payouts: pd.DataFrame, models: tuple[str, ...],
                          rule_value: str = "Trot G", n_boot: int = 4000,
                          seed: int = 0) -> dict[str, dict[str, dict[str, Any]]]:
    """Règle figée (par défaut « 3e du modèle, gagnant, trot G ») appliquée avec chaque
    modèle, sur l'année de validation (≤ FROZEN_SPLIT), l'année de découverte et le total."""
    from datetime import date as _date

    from horse_intelligence.evaluation.frozen_rules import FROZEN_RULES, rule_bets

    rule = next(r for r in FROZEN_RULES if r.value == rule_value)
    split = _date.fromisoformat(FROZEN_SPLIT)
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model in models:
        bets = rule_bets(frame, payouts, model, rule)
        periods = {"validation": bets[bets["race_date"] <= split],
                   "découverte": bets[bets["race_date"] > split], "total": bets}
        out[model] = {}
        for name, b in periods.items():
            p = b["profit"].to_numpy()
            if len(p) >= 2:
                boots = p[rng.integers(0, len(p), size=(n_boot, len(p)))].mean(1)
                ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
            else:
                ci = (float("nan"), float("nan"))
            out[model][name] = {"n_bets": len(p), "roi": float(p.mean()) if len(p) else
                                float("nan"), "hit_rate": float((p > -1).mean()) if len(p)
                                else float("nan"), "roi_ci": ci}
    return out


MODEL_CHALLENGERS = ("rank_market", "catboost_market", "stack_market")


def models_runners(catboost_refit_days: int = 14,
                   fits: list[dict[str, Any]] | None = None) -> dict[str, ModelRunner]:
    """LightGBM de production et candidats, appris UNE fois par pli et partagés :
    `lgbm_market` (référence actuelle), `rank_market` (classement par course),
    `catboost_market`, `stack_market` (logit conditionnel sur les trois scores + marché).
    CatBoost, plus lent, n'est réappris que tous les `catboost_refit_days` jours de données
    (sans fuite : un modèle appris sur un pli ne voit que des dates antérieures)."""
    from horse_intelligence.models.challengers import (
        CalibratedScores,
        CatBoostScore,
        LgbmBinaryScore,
        LgbmRankScore,
    )

    log = fits if fits is not None else []
    state: dict[str, Any] = {"key": None, "cat": None, "cat_until": None}

    def bank(base: pd.DataFrame) -> dict[str, Any]:
        key = (base["race_date"].max(), len(base))
        if state["key"] != key:
            t0 = time.perf_counter()
            last = base["race_date"].max()
            state["lgbm"] = LgbmBinaryScore(LGBM_FEATURES, CATEGORICAL).fit(base)
            state["rank"] = LgbmRankScore(LGBM_FEATURES, CATEGORICAL).fit(base)
            refit = state["cat"] is None or (last - state["cat_until"]).days >= catboost_refit_days
            if refit:
                state["cat"] = CatBoostScore(LGBM_FEATURES, CATEGORICAL).fit(base)
                state["cat_until"] = last
            state["key"] = key
            log.append({"trained_until": str(last), "catboost_refit": refit,
                        "seconds": round(time.perf_counter() - t0, 1)})
        return state

    def runner(names: tuple[str, ...]) -> ModelRunner:
        def run(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
            b = bank(base)
            m = CalibratedScores({n: b[n] for n in names}, with_market=True)
            m.fit(_with_market(calib))
            sub = _with_market(test)
            return _masked(test, sub, m.predict(sub))
        return run

    return {"lgbm_market": runner(("lgbm",)), "rank_market": runner(("rank",)),
            "catboost_market": runner(("cat",)),
            "stack_market": runner(("lgbm", "rank", "cat"))}


def run_models_experiment(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                          calib_days: int = 7, catboost_refit_days: int = 14) -> dict[str, Any]:
    """D'autres modèles que LightGBM binaire font-ils mieux ? Critère fixé AVANT de voir les
    résultats : un des candidats bat `lgbm_market` en log-loss, IC 95 % entièrement < 0."""
    t0 = time.perf_counter()
    raw = load_raw(engine)
    ds = build_dataset(raw)
    if not audit_source_aggregates(ds.frame).passed:
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]
    fits: list[dict[str, Any]] = []
    runners = {REFERENCE_MODEL: model_runners()[REFERENCE_MODEL],
               **models_runners(catboost_refit_days, fits)}
    wf = run_walk_forward(frame, runners, min_train_days, test_days, calib_days)
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = list(runners)
    mkt = _with_market(evald)
    mkt = mkt[mkt.groupby("race_id")[names].transform(lambda s: s.notna().all()).all(axis=1)]
    vs_inc = {n: paired_bootstrap(mkt, mkt[n], mkt[INCUMBENT]) for n in MODEL_CHALLENGERS}
    dates = sorted(frame["race_date"].unique())
    return {
        "period": [str(dates[0]), str(dates[-1])], "n_races": int(mkt["race_id"].nunique()),
        "n_folds": len(wf.folds), "seconds": round(time.perf_counter() - t0, 1),
        "metrics": {n: compute_metrics(mkt, mkt[n]).as_dict() for n in names},
        "vs_reference": {n: paired_bootstrap(mkt, mkt[n], mkt[REFERENCE_MODEL])
                         for n in names if n != REFERENCE_MODEL},
        "vs_incumbent": vs_inc,
        "success": any(d["ci_high"] < 0 for d in vs_inc.values()),
        "fits": fits[:3] + fits[-2:], "n_fits": len(fits),
        "catboost_refits": sum(f["catboost_refit"] for f in fits),
        "errors": wf.errors,
    }


PLACE_INCUMBENT = "harville_lgbm"
PLACE_CHALLENGERS = ("place_market", "place_stack")
PLACE_MODELS = ("harville_market", "harville_lgbm", "harville_lgbm_cal", *PLACE_CHALLENGERS)


def place_runners() -> dict[str, ModelRunner]:
    """Probabilités de PLACE (et non de victoire), apprises une fois par pli et partagées :
    - `harville_market` : Harville sur les cotes du marché ;
    - `harville_lgbm` : Harville sur `lgbm_market` (méthode de production) ;
    - `harville_lgbm_cal` : la même, recalibrée (logistique + ajustement par course) ;
    - `place_market` : LightGBM « placé » combiné à Harville du marché ;
    - `place_stack` : LightGBM « placé » combiné aux deux Harville.
    Logistique de combinaison apprise sur la partie récente de l'apprentissage."""
    from horse_intelligence.models.place_model import (
        PlaceCombiner,
        PlaceModel,
        harville_place,
    )

    state: dict[str, Any] = {"key": None}

    def inputs(sub: pd.DataFrame, win: LgbmRaceModel, place: PlaceModel) -> dict[str, Any]:
        return {"h_mkt": harville_place(sub, sub["mkt_prob"]),
                "h_lgbm": harville_place(sub, win.predict(sub)),
                "score": place.raw_score(sub)}

    def bank(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        key = (base["race_date"].max(), len(base), test["race_date"].min())
        if state["key"] != key:
            cal, sub = _with_market(calib), _with_market(test)
            win = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL, with_market=True).fit(base, cal)
            place = PlaceModel(LGBM_FEATURES, CATEGORICAL).fit_score(base)
            state.update(key=key, cal=cal, sub=sub, ci=inputs(cal, win, place),
                         ti=inputs(sub, win, place))
        return state

    def lg(p: FloatArray) -> FloatArray:
        q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.asarray(np.log(q / (1 - q)), dtype=np.float64)

    def fixed(col: str) -> ModelRunner:
        def run(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
            b = bank(base, calib, test)
            return _masked(test, b["sub"], b["ti"][col])
        return run

    def combined(cols: tuple[str, ...]) -> ModelRunner:
        def matrix(d: dict[str, Any]) -> FloatArray:
            return np.column_stack([d[c] if c == "score" else lg(d[c]) for c in cols])

        def run(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
            b = bank(base, calib, test)
            m = PlaceCombiner().fit(matrix(b["ci"]), b["cal"])
            return _masked(test, b["sub"], m.predict(matrix(b["ti"]), b["sub"]))
        return run

    return {"harville_market": fixed("h_mkt"), "harville_lgbm": fixed("h_lgbm"),
            "harville_lgbm_cal": combined(("h_lgbm",)),
            "place_market": combined(("score", "h_mkt")),
            "place_stack": combined(("score", "h_lgbm", "h_mkt"))}


def place_paired_bootstrap(frame: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    from horse_intelligence.models.place_model import place_paired_bootstrap as boot

    return boot(frame, frame[a], frame[b])


def place_metrics(frame: pd.DataFrame, col: str) -> dict[str, float]:
    from horse_intelligence.evaluation.metrics import auc_score
    from horse_intelligence.models.place_model import place_log_losses, placed_target

    y, p = placed_target(frame), frame[col]
    long = frame["mkt_odds"] > 10
    top = frame.assign(_p=p).sort_values(["race_id", "_p"], ascending=[True, False]) \
        .drop_duplicates("race_id")
    return {"log_loss": float(place_log_losses(frame, p).mean()),
            "brier": float(((p - y) ** 2).groupby(frame["race_id"]).sum().mean()),
            "auc": auc_score(y, p),
            "top1_placed": float(placed_target(top).mean()),
            "outsiders_pred": float(p[long].mean()), "outsiders_obs": float(y[long].mean())}


def run_place_experiment(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                         calib_days: int = 7) -> dict[str, Any]:
    """Un modèle appris directement sur « placé » fait-il mieux que Harville appliqué au
    modèle gagnant (méthode de production) ? Critère fixé AVANT de voir les résultats :
    `place_market` ou `place_stack` bat `harville_lgbm` en log-loss « placé », IC 95 %
    entièrement < 0. Puis sélection / test de règles de simple placé."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.place_betting import select_then_test
    from horse_intelligence.models.place_model import place_eval_frame

    t0 = time.perf_counter()
    raw = load_raw(engine)
    ds = build_dataset(raw)
    if not audit_source_aggregates(ds.frame).passed:
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]
    runners = place_runners()
    wf = run_walk_forward(frame, runners, min_train_days, test_days, calib_days)
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = list(runners)
    f = place_eval_frame(evald, names)
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    vs_inc = {n: place_paired_bootstrap(f, n, PLACE_INCUMBENT)
              for n in ("harville_lgbm_cal", *PLACE_CHALLENGERS)}
    dates = sorted(frame["race_date"].unique())
    return {
        "period": [str(dates[0]), str(dates[-1])], "n_races": int(f["race_id"].nunique()),
        "n_folds": len(wf.folds), "seconds": round(time.perf_counter() - t0, 1),
        "metrics": {n: place_metrics(f, n) for n in names},
        "vs_market": {n: place_paired_bootstrap(f, n, "harville_market")
                      for n in names if n != "harville_market"},
        "vs_incumbent": vs_inc,
        "success": any(vs_inc[c]["ci_high"] < 0 for c in PLACE_CHALLENGERS),
        "betting": {n: select_then_test(f, payouts, n)
                    for n in ("harville_market", "harville_lgbm", *PLACE_CHALLENGERS)},
        "errors": wf.errors,
    }
