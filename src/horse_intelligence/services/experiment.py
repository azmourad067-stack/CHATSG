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
    d'origine contre v1 + v2 ; `stage="v3"` : v1 + v2 contre + v3 ; `stage="v4"` : production
    contre + v4 (origines, valeur handicap). Critère fixé
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
    ds = build_dataset(raw, with_v4=stage == "v4")
    if not audit_source_aggregates(ds.frame).passed:
        ds = build_dataset(raw, use_source_aggregates=False, with_v4=stage == "v4")
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]
    base_runners = model_runners()
    importances: dict[str, pd.Series] = {}
    # v2 : variables d'origine contre v1 + v2 ; v3 : v1 + v2 contre v1 + v2 + v3
    from horse_intelligence.intelligence.feature_engineering import V4_FEATURES

    stages = {"v2": (LGBM_FEATURES_V1, LGBM_FEATURES_V2), "v3": (LGBM_FEATURES_V2, LGBM_FEATURES),
              # v4 : production contre production + origines et valeur handicap
              "v4": (LGBM_FEATURES, [*LGBM_FEATURES, *V4_FEATURES])}
    old_feats, new_feats = stages[stage]
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
                          seed: int = 0, bet_type: str | None = None
                          ) -> dict[str, dict[str, dict[str, Any]]]:
    """Règle figée (par défaut « 3e du modèle, gagnant, trot G ») appliquée avec chaque
    modèle, sur l'année de validation (≤ FROZEN_SPLIT), l'année de découverte et le total."""
    from datetime import date as _date

    from horse_intelligence.evaluation.frozen_rules import FROZEN_RULES, rule_bets

    rule = next(r for r in FROZEN_RULES if r.value == rule_value)
    if bet_type is not None:  # même règle jouée sur un autre pari (ex. placé)
        rule = replace(rule, bet_type=bet_type)
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


def run_betting2_experiment(engine: Engine, model_col: str = INCUMBENT,
                            place_col: str = "place_market") -> dict[str, Any]:
    """Deux pistes de paris sur les prédictions hors échantillon du dernier entraînement.

    1. Paris combinés (couplé gagnant / placé, trio, 2 sur 4) avec des probabilités de
       paire déduites du MODÈLE PLACÉ (Plackett-Luce calé sur P(placé)), comparées à
       l'approche précédente (Plackett-Luce sur les probabilités de victoire).
    2. Espérance calculée avec le RAPPORT FINAL PRÉVU (modèle appris mois par mois sur les
       courses passées) au lieu de la cote de référence, en simple gagnant et placé.

    Critère fixé AVANT de voir les résultats : une règle choisie sur la 1re moitié est
    rentable sur la 2nde avec un IC 95 % du ROI entièrement > 0."""
    from sqlalchemy import text

    from horse_intelligence.evaluation import payout_model as pm
    from horse_intelligence.evaluation.combo_simulator import (
        COMBO_BETS,
        build_candidates,
        select_then_test_combos,
    )
    from horse_intelligence.models.place_model import pl_place_column
    from horse_intelligence.services.training import load_oos_predictions

    t0 = time.perf_counter()
    pred = load_oos_predictions(engine)
    if pred is None or place_col not in pred:
        raise ValueError("prédictions hors échantillon du modèle placé absentes : "
                         "relancez l'entraînement")
    pred = _with_market(pred)
    pred = pred[pred.groupby("race_id")[[model_col, place_col, REFERENCE_MODEL]]
                .transform(lambda s: s.notna().all()).all(axis=1)].copy()
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    # --- 1. paris combinés
    pred["pl_place"] = pl_place_column(pred, place_col)
    combos: dict[str, dict[str, Any]] = {}
    for col in (model_col, "pl_place"):
        cand = build_candidates(pred.dropna(subset=[col]), payouts, col, REFERENCE_MODEL).table
        combos[col] = {bt: asdict(select_then_test_combos(cand, bt)) for bt in COMBO_BETS
                       if not cand.empty and (cand["bet_type"] == bt).any()}
    t_combos = round(time.perf_counter() - t0, 1)
    # --- 2. rapport final prévu
    f = pm.predict_payouts(pm.attach_payouts(pred, payouts), model_col, place_col)
    singles = {
        "gagnant (cote de référence)": pm.select_then_test(f, "win", "reference", model_col,
                                                           place_col),
        "gagnant (rapport prévu)": pm.select_then_test(f, "win", "predicted", model_col,
                                                       place_col),
        "placé (rapport prévu)": pm.select_then_test(f, "place", "predicted", model_col,
                                                     place_col),
    }
    tests = [r["test"] for r in singles.values()] + [
        r["test"] for by_bt in combos.values() for r in by_bt.values()]
    return {
        "period": [str(pred["race_date"].min()), str(pred["race_date"].max())],
        "n_races": int(pred["race_id"].nunique()),
        "seconds": round(time.perf_counter() - t0, 1), "seconds_combos": t_combos,
        "combos": combos, "payout_accuracy": pm.payout_accuracy(f), "singles": singles,
        "success": any(t is not None and t["n_bets"] > 0 and t["roi_ci"][0] > 0
                       for t in tests),
    }


TUNING_CONFIGS: dict[str, tuple[dict[str, Any], int]] = {
    # nom -> (paramètres LightGBM modifiés par rapport à DEFAULT_PARAMS, nombre d'arbres) ;
    # « actuel » = réglage de production jusqu'au 15/11/2026 (le gagnant « plus_profond »
    # est depuis RACE_PARAMS / RACE_NUM_ROUNDS)
    "actuel": ({}, 300),
    "plus_profond": ({"num_leaves": 31, "min_data_in_leaf": 60}, 400),
    "plus_lent": ({"learning_rate": 0.015}, 600),
    "regularise": ({"min_data_in_leaf": 150, "lambda_l2": 20.0, "feature_fraction": 0.6},
                   400),
    "large": ({"num_leaves": 63, "learning_rate": 0.02, "min_data_in_leaf": 100,
               "feature_fraction": 0.7}, 500),
}


# Second test (15/11/2026) : « large » contre le réglage de production « plus_profond ».
# Les deux moitiés ont déjà été vues : critère plus strict, IC 95 % < 0 sur CHAQUE moitié.
TUNING_LARGE_CONFIGS: dict[str, tuple[dict[str, Any], int]] = {
    "production": ({"num_leaves": 31, "min_data_in_leaf": 60}, 400),
    "large": ({"num_leaves": 63, "learning_rate": 0.02, "min_data_in_leaf": 100,
               "feature_fraction": 0.7}, 500),
}


def run_tuning_experiment(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                          calib_days: int = 7,
                          configs: dict[str, tuple[dict[str, Any], int]] | None = None
                          ) -> dict[str, Any]:
    """Réglage de LightGBM (modèle « + marché ») sans tricher : le meilleur réglage est
    choisi sur la 1re moitié des plis, puis jugé sur la 2nde. Critère fixé AVANT de voir les
    résultats : sur la 2nde moitié, le réglage choisi bat le réglage actuel, IC 95 % < 0."""
    from horse_intelligence.models.lightgbm_model import DEFAULT_PARAMS

    cfgs = configs if configs is not None else TUNING_CONFIGS
    t0 = time.perf_counter()
    raw = load_raw(engine)
    ds = build_dataset(raw)
    if not audit_source_aggregates(ds.frame).passed:
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]

    def runner(params: dict[str, Any], rounds: int) -> ModelRunner:
        def run(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
            m = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL, with_market=True,
                              params={**DEFAULT_PARAMS, "num_threads": 4, **params},
                              num_rounds=rounds)
            m.fit(base, _with_market(calib))
            sub = _with_market(test)
            return _masked(test, sub, m.predict(sub))
        return run

    runners = {name: runner(p, r) for name, (p, r) in cfgs.items()}
    wf = run_walk_forward(frame, runners, min_train_days, test_days, calib_days)
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = list(runners)
    mkt = _with_market(evald)
    mkt = mkt[mkt.groupby("race_id")[names].transform(lambda s: s.notna().all()).all(axis=1)]
    half = len(wf.folds) // 2
    first, second = mkt[mkt["fold"] < half], mkt[mkt["fold"] >= half]
    base_name = next(iter(cfgs))
    on_first = {n: paired_bootstrap(first, first[n], first[base_name])
                for n in names if n != base_name}
    chosen = min(on_first, key=lambda n: on_first[n]["mean_diff"])
    on_second = {n: paired_bootstrap(second, second[n], second[base_name])
                 for n in names if n != base_name}
    return {
        "n_races": int(mkt["race_id"].nunique()), "n_folds": len(wf.folds),
        "seconds": round(time.perf_counter() - t0, 1),
        "selection_period": [str(first["race_date"].min()), str(first["race_date"].max())],
        "test_period": [str(second["race_date"].min()), str(second["race_date"].max())],
        "configs": {n: {"params": p, "num_rounds": r} for n, (p, r) in cfgs.items()},
        "first_half": on_first, "second_half": on_second, "chosen": chosen,
        "success": on_second[chosen]["ci_high"] < 0,
        # critère plus strict (réglage déjà vu sur les deux moitiés) : meilleur sur CHACUNE
        "success_both_halves": on_first[chosen]["ci_high"] < 0
        and on_second[chosen]["ci_high"] < 0,
        "metrics_second_half": {n: compute_metrics(second, second[n]).as_dict() for n in names},
        "errors": wf.errors,
    }


def run_frozen_by_audience(engine: Engine, model_col: str = INCUMBENT,
                           rule_value: str = "Trot G") -> dict[str, Any]:
    """Règle figée (par défaut « 3e du modèle, gagnant, trot G ») sur les prédictions hors
    échantillon du dernier entraînement, séparément pour les réunions NATIONALES (programme
    de pmu.fr, où l'on peut parier en ligne) et RÉGIONALES (LOCAL / REGIONAL)."""
    from sqlalchemy import text

    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez l'entraînement")
    with engine.connect() as conn:
        races = pd.read_sql(text("SELECT race_id, label, conditions_text, reunion_num, "
                                 "audience FROM races"), conn)
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    frame = _with_market(pred).merge(races, on="race_id", how="left")
    frame["groupe_audience"] = np.where(frame["audience"].isin(["LOCAL", "REGIONAL"]),
                                        "régionales", np.where(frame["audience"].isna(),
                                                               "inconnue", "nationales"))
    out: dict[str, Any] = {"period": [str(frame["race_date"].min()),
                                      str(frame["race_date"].max())],
                           "n_races": {g: int(f["race_id"].nunique())
                                       for g, f in frame.groupby("groupe_audience")}}
    subsets = {"toutes": frame,
               "nationales": frame[frame["groupe_audience"] == "nationales"],
               "régionales": frame[frame["groupe_audience"] == "régionales"]}
    out["rule"] = {name: frozen_rule_by_period(f, payouts, (model_col,), rule_value)[model_col]
                   for name, f in subsets.items() if not f.empty}
    # même cheval (3e du modèle) joué en simple placé (demande du 16/11/2026)
    out["rule_place"] = {name: frozen_rule_by_period(f, payouts, (model_col,), rule_value,
                                                     bet_type="SIMPLE_PLACE")[model_col]
                         for name, f in subsets.items() if not f.empty}
    return out


TIERCE_STRATEGIES = ("6 premiers du modèle (combiné)", "20 trios les plus probables",
                     "6 premiers du marché (repère)")


def _quinte_frame(engine: Engine, cols: list[str]
                  ) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Prédictions hors échantillon des courses du quinté (marché complet), rapports
    Internet, ensemble des courses du quinté en base."""
    from sqlalchemy import text

    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez l'entraînement")
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts WHERE pool = 'INTERNET'"), conn)
    quinte = set(payouts.loc[payouts["bet_type"] == "QUINTE_PLUS", "race_id"].astype(int))
    f = pred[pred["race_id"].isin(quinte)]
    if f.empty:
        return f, payouts, quinte
    f = _with_market(f)
    cols = [c for c in cols if c in f]
    f = f[f.groupby("race_id")[cols].transform(lambda s: s.notna().all()).all(axis=1)].copy()
    return f, payouts, quinte


def run_tierce_quinte(engine: Engine, model_col: str = INCUMBENT,
                      place_col: str = "place_market") -> dict[str, Any]:
    """20 combinaisons de tiercé par course du quinté (pas de trio sur cette course), sur
    les prédictions hors échantillon du dernier entraînement, réglées aux rapports officiels
    ordre / désordre. Critère fixé AVANT de voir les résultats : une des deux stratégies du
    modèle est rentable sur la 2nde moitié, IC 95 % du ROI entièrement > 0."""
    from horse_intelligence.evaluation import tierce_quinte as tq
    from horse_intelligence.models.place_model import pl_place_column

    f, payouts, quinte = _quinte_frame(engine, [model_col, "mkt_prob", place_col])
    strengths = place_col if place_col in f else model_col
    if f.empty:
        return {"n_quinte": len(quinte), "n_quinte_predicted": 0, "n_quinte_settled": 0,
                "strengths": strengths, "results": {}, "success": False}
    f["_pl"] = pl_place_column(f, place_col) if place_col in f else f[model_col]
    paid = tq.tierce_results(payouts)
    rng = np.random.default_rng(0)

    def nums(g: pd.DataFrame) -> np.ndarray:
        return g["num_pmu"].to_numpy()

    makers = {
        TIERCE_STRATEGIES[0]: lambda g: tq.top6_tickets(nums(g), g[model_col].to_numpy()),
        TIERCE_STRATEGIES[1]: lambda g: tq.proba20_tickets(
            nums(g), g["_pl"].to_numpy(), g[model_col].to_numpy(), rng),
        TIERCE_STRATEGIES[2]: lambda g: tq.top6_tickets(nums(g), g["mkt_prob"].to_numpy()),
    }
    results = {name: tq.by_period(tq.simulate(f, paid, make)) for name, make in makers.items()}
    return {
        "n_quinte": len(quinte), "n_quinte_predicted": int(f["race_id"].nunique()),
        "n_quinte_settled": len(set(f["race_id"]) & set(paid)),
        "strengths": strengths, "results": results,
        "success": any(results[s].get("2nde moitié", {}).get("roi_ci", (0, 0))[0] > 0
                       for s in TIERCE_STRATEGIES[:2]),
    }


QUINTE_FAMILIES = {
    "combiné du modèle": ("combine", "model", (5, 6, 7, 8, 9, 10)),
    "1 base du modèle + N": ("base", "model", (4, 5, 6, 7, 8, 9)),
    "combiné du marché (repère)": ("combine", "market", (5, 6, 7, 8, 9, 10)),
}


def run_quinte_plus(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Quinté+ : nombre de combinaisons contre gain, sur les prédictions hors échantillon,
    réglé aux rapports officiels (ordre, désordre, bonus 4 / 4sur5, bonus 3).

    Critère fixé AVANT de voir les résultats : pour chaque formule du modèle, le nombre de
    chevaux N est choisi sur la 1re moitié (meilleur ROI) ; succès si, pour ce N, l'IC 95 %
    du ROI de la 2nde moitié est entièrement > 0."""
    from horse_intelligence.evaluation import quinte_plus as qp
    from horse_intelligence.evaluation import tierce_quinte as tq

    f, payouts, quinte = _quinte_frame(engine, [model_col, "mkt_prob"])
    out: dict[str, Any] = {"n_quinte": len(quinte), "n_quinte_predicted": 0,
                           "n_quinte_settled": 0, "families": {}, "chosen": {},
                           "success": False}
    if f.empty:
        return out
    paid = qp.quinte_results(payouts)
    out["n_quinte_predicted"] = int(f["race_id"].nunique())
    out["n_quinte_settled"] = len(set(f["race_id"]) & set(paid))
    makers = {"combine": qp.combined_tickets, "base": qp.one_base_tickets}
    for name, (formula, source, sizes) in QUINTE_FAMILIES.items():
        col = model_col if source == "model" else "mkt_prob"
        fam: dict[int, dict[str, Any]] = {}
        for n in sizes:
            def make(g: pd.DataFrame, n: int = n, formula: str = formula,
                     col: str = col) -> list[tuple[int, ...]]:
                return makers[formula](g["num_pmu"].to_numpy(), g[col].to_numpy(), n)

            res = qp.simulate(f, paid, make)
            fam[n] = {"n_tickets": qp.n_tickets(formula, n), "periods": tq.by_period(res),
                      "ranks": qp.rank_rates(res)}
        out["families"][name] = fam
        if source == "model":
            best = max(sizes, key=lambda n: fam[n]["periods"].get("1re moitié", {})
                       .get("roi", -np.inf))
            out["chosen"][name] = best
    out["success"] = any(
        out["families"][name][n]["periods"].get("2nde moitié", {}).get("roi_ci", (0, 0))[0] > 0
        for name, n in out["chosen"].items())
    return out


def run_profile_search(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Profils de course (rang du modèle × cote × discipline × partants × avantage) pour le
    simple gagnant / placé : sélection sur la 1re moitié, jugement sur la 2nde avec un IC
    corrigé pour le nombre de profils retenus (protocole fixé dans profile_search)."""
    from sqlalchemy import text

    from horse_intelligence.evaluation import profile_search as ps
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez l'entraînement")
    pred = _with_market(pred)
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, pool, bet_type, combination, payout_per_eur FROM payouts "
            "WHERE pool = 'INTERNET' AND bet_type IN ('SIMPLE_GAGNANT', 'SIMPLE_PLACE')"),
            conn)
    f = ps.prepare(pred, payouts, model_col)
    f = f[f["profit_gagnant"].notna() | f["profit_placé"].notna()]
    if f.empty:
        return {"n_races": 0, "success": False, "tested": []}
    return {"n_races": int(f["race_id"].nunique()), **ps.search(f)}


def run_grand_search(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Grande recherche (simples + combinés) en trois périodes : découverte, validation,
    test final jamais utilisé pour choisir (protocole fixé dans grand_search)."""
    from sqlalchemy import text

    from horse_intelligence.evaluation import grand_search as gs
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez l'entraînement")
    pred = _with_market(pred)
    pred = pred[pred[model_col].notna() & pred["mkt_odds"].notna()]
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, pool, bet_type, combination, payout_per_eur FROM payouts "
            "WHERE pool = 'INTERNET' AND bet_type IN ('SIMPLE_GAGNANT', 'SIMPLE_PLACE', "
            "'COUPLE_GAGNANT', 'COUPLE_PLACE', 'DEUX_SUR_QUATRE', 'TRIO')"), conn)
    if pred.empty:
        return {"n_races": 0, "success": False, "finalists": []}
    cuts = gs.cuts_of(pred["race_date"])
    from horse_intelligence.evaluation.profile_search import prepare

    f = prepare(pred, payouts, model_col)
    f = f[f["profit_gagnant"].notna() | f["profit_placé"].notna()]
    races = gs.combo_races(pred, payouts, model_col)
    s_str, s_cnt, s_gain = gs.simple_scan(f, model_col, cuts)
    strategies, counts, gains = s_str, s_cnt, s_gain
    if not races.empty:
        c_str, c_cnt, c_gain = gs.combo_scan(races, cuts)
        strategies = [*s_str, *c_str]
        counts, gains = np.vstack([s_cnt, c_cnt]), np.vstack([s_gain, c_gain])
    sel = gs.select(strategies, counts, gains)
    alpha = 0.05 / max(1, len(sel["final"]))

    def bets(i: int) -> pd.DataFrame:
        s = strategies[i]
        return gs.simple_bets(f, model_col, s) if s.family == "simple" \
            else gs.combo_bets(races, s)

    def report(i: int, final_alpha: float = 0.05) -> dict[str, Any]:
        b = bets(i)
        per = gs.periods(b["race_date"], cuts) if not b.empty else np.array([], int)
        return {"label": strategies[i].label, "family": strategies[i].family,
                "A": gs.evaluate(b[per == 0]), "B": gs.evaluate(b[per == 1]),
                "C": gs.evaluate(b[per == 2], alpha=final_alpha)}

    finalists = [report(i, alpha) for i in sel["final"]]
    return {
        "n_races": int(pred["race_id"].nunique()),
        "period": (str(pred["race_date"].min()), str(pred["race_date"].max())),
        "cuts": (str(cuts[0]), str(cuts[1])),
        "n_strategies": sel["n_strategies"], "n_simple": len(s_str),
        "n_eligible": sel["n_eligible"], "n_discovery": len(sel["discovery"]),
        "n_validated": len(sel["validated"]), "alpha": alpha, "chance": sel["chance"],
        "top_discovery": [report(i) for i in sel["discovery"][:5]],
        "finalists": finalists,
        "success": any(r["C"].get("roi_ci", (0, 0))[0] > 0 for r in finalists),
    }


def run_value46_bet_types(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Règle figée « Valeur 4e-6e » en simple gagnant, en simple placé et les deux, sur les
    trois périodes de la grande recherche (A, B, C) et au total. Descriptif : la règle
    figée reste en gagnant ; une version placé serait une nouvelle piste à suivre à part."""
    from sqlalchemy import text

    from horse_intelligence.evaluation import grand_search as gs
    from horse_intelligence.evaluation.profile_search import prepare
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez l'entraînement")
    pred = _with_market(pred)
    pred = pred[pred[model_col].notna() & pred["mkt_odds"].notna()]
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, pool, bet_type, combination, payout_per_eur FROM payouts "
            "WHERE pool = 'INTERNET' AND bet_type IN ('SIMPLE_GAGNANT', 'SIMPLE_PLACE')"),
            conn)
    if pred.empty:
        return {"n_races": 0, "results": {}}
    cuts = gs.cuts_of(pred["race_date"])
    f = prepare(pred, payouts, model_col)
    # mêmes chevaux pour les deux paris : courses où gagnant ET placé ont été réglés
    f = f[f["profit_gagnant"].notna() & f["profit_placé"].notna()]
    out: dict[str, Any] = {}
    bets = {}
    for bet in ("gagnant", "placé"):
        key = ((4, 6), (9.0, 20.0), None, (10, 14), "espérance ≥ +10 %", bet)
        bets[bet] = gs.simple_bets(f, model_col, gs.Strategy("simple", bet, key))
    both = pd.concat([bets["gagnant"], bets["placé"]]).groupby(
        ["race_id", "race_date"], as_index=False)[["stake", "profit"]].sum()
    for name, b in (("gagnant", bets["gagnant"]), ("placé", bets["placé"]),
                    ("gagnant + placé", both)):
        per = gs.periods(b["race_date"], cuts) if not b.empty else np.array([], int)
        out[name] = {"A": gs.evaluate(b[per == 0]), "B": gs.evaluate(b[per == 1]),
                     "C": gs.evaluate(b[per == 2]), "total": gs.evaluate(b)}
    return {"n_races": int(f["race_id"].nunique()),
            "cuts": (str(cuts[0]), str(cuts[1])),
            "period": (str(f["race_date"].min()), str(f["race_date"].max())),
            "results": out}


ANGLE_COLUMNS = ["shoes_d4_first", "shoes_d4", "blinkers_change", "blinkers_on",
                 "rider_change", "jockey_win_rate_30d", "jockey_runs_30d",
                 "trainer_win_rate_30d", "trainer_runs_30d", "class_change",
                 "horse_days_since_db", "mus_last_pos", "apt_course_runs", "apt_course_pct",
                 "apt_dist_runs", "apt_dist_pct", "apt_pair_runs", "apt_pair_pct",
                 "speed_red_mean3", "trainer_course_runs", "trainer_course_win_rate",
                 "dist_change"]


def _features_frame(engine: Engine, model_col: str, columns: list[str]
                    ) -> tuple[pd.DataFrame, list[str]]:
    """Prédictions hors échantillon (marché complet) réglées en simple gagnant, avec les
    variables demandées (calculées sans fuite). Renvoie aussi les variables absentes."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.profile_search import prepare
    from horse_intelligence.intelligence.feature_engineering import build_dataset, load_raw
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez l'entraînement")
    pred = _with_market(pred)
    pred = pred[pred[model_col].notna() & pred["mkt_odds"].notna()]
    if pred.empty:
        return pred, columns
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, pool, bet_type, combination, payout_per_eur FROM payouts "
            "WHERE pool = 'INTERNET' AND bet_type = 'SIMPLE_GAGNANT'"), conn)
    frame = build_dataset(load_raw(engine)).frame
    cols = [c for c in columns if c in frame.columns and c not in ("race_id", "num_pmu")]
    feats = frame[["race_id", "num_pmu", *cols]]
    del frame
    f = prepare(pred, payouts, model_col)
    f = f[f["profit_gagnant"].notna()].merge(feats, on=["race_id", "num_pmu"], how="left")
    f["_p_model"] = f[model_col]
    return f, sorted(set(columns) - set(cols))


def run_angles(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """« Angles » de turfiste en simple gagnant (evaluation/angles.py), sur les prédictions
    hors échantillon et les variables calculées sans fuite."""
    from horse_intelligence.evaluation import angles

    f, missing = _features_frame(engine, model_col, ANGLE_COLUMNS)
    if f.empty:
        return {"n_races": 0, "success": False, "finalists": []}
    rep = angles.run(f)
    return {"n_races": int(f["race_id"].nunique()),
            "period": (str(f["race_date"].min()), str(f["race_date"].max())),
            "missing_columns": missing, **rep}


def run_d4_draw(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Déferré des 4 pour la 1re fois × numéro de corde (evaluation/d4_draw.py)."""
    from horse_intelligence.evaluation import d4_draw

    f, missing = _features_frame(engine, model_col, ["shoes_d4_first", "draw", "autostart"])
    if f.empty or "shoes_d4_first" in missing:
        return {"n_races": 0, "success": False, "finalists": [], "missing_columns": missing}
    if "autostart" in missing:
        f["autostart"] = np.nan
    rep = d4_draw.run(f)
    return {"n_races": int(f["race_id"].nunique()),
            "period": (str(f["race_date"].min()), str(f["race_date"].max())),
            "missing_columns": missing, **rep}


def run_context_search(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Recherche par contexte de course (evaluation/context_search.py), simple gagnant et
    placé, sur les prédictions hors échantillon."""
    from horse_intelligence.evaluation import context_search

    f = _context_frame(engine, model_col)
    if f.empty:
        return {"n_races": 0, "success": False, "finalists": []}
    rep = context_search.run(f, model_col)
    return {"n_races": int(f["race_id"].nunique()),
            "period": (str(f["race_date"].min()), str(f["race_date"].max())), **rep}


def _context_frame(engine: Engine, model_col: str) -> pd.DataFrame:
    """Partants hors échantillon réglés en gagnant ou en placé, avec le contexte de la
    course (réunion, libellé, conditions, allocation, conditions de sexe)."""
    from sqlalchemy import text

    from horse_intelligence.evaluation.profile_search import prepare
    from horse_intelligence.services.training import load_oos_predictions

    pred = load_oos_predictions(engine)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez l'entraînement")
    pred = _with_market(pred)
    pred = pred[pred[model_col].notna() & pred["mkt_odds"].notna()]
    if pred.empty:
        return pred
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, pool, bet_type, combination, payout_per_eur FROM payouts "
            "WHERE pool = 'INTERNET' AND bet_type IN ('SIMPLE_GAGNANT', 'SIMPLE_PLACE')"),
            conn)
        races = pd.read_sql(text(
            "SELECT race_id, audience, label, conditions_text, prize_eur, sex_condition "
            "FROM races"), conn)
    f = prepare(pred, payouts, model_col)
    f = f[f["profit_gagnant"].notna() | f["profit_placé"].notna()]
    return f.drop(columns=[c for c in ("audience", "label", "conditions_text", "prize_eur",
                                       "sex_condition") if c in f]).merge(
        races, on="race_id", how="left")


def run_market_bias(engine: Engine, model_col: str = INCUMBENT) -> dict[str, Any]:
    """Carte du marché par contexte et stratégies regroupées (evaluation/market_bias.py)."""
    from horse_intelligence.evaluation import market_bias

    f = _context_frame(engine, model_col)
    if f.empty:
        return {"n_races": 0, "success": False}
    rep = market_bias.run(f, model_col)
    return {"n_races": int(f["race_id"].nunique()),
            "period": (str(f["race_date"].min()), str(f["race_date"].max())), **rep}


def _corrected_paired(frame: pd.DataFrame, a: str, b: str, alpha: float, n_boot: int = 2000,
                      seed: int = 0) -> dict[str, float]:
    """Différence de log-loss par course (a − b), IC à 1 − alpha (rééchantillonnage des
    courses). Négatif = a meilleur."""
    from horse_intelligence.evaluation.metrics import eval_frame, race_log_losses

    la = race_log_losses(eval_frame(frame, frame[a]))
    lb = race_log_losses(eval_frame(frame, frame[b]))
    common = la.index.intersection(lb.index)
    d = (la.loc[common] - lb.loc[common]).to_numpy()
    if d.size == 0:
        return {"n_races": 0, "mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return {"n_races": int(d.size), "mean_diff": float(d.mean()),
            "ci_low": float(np.quantile(boots, alpha / 2)),
            "ci_high": float(np.quantile(boots, 1 - alpha / 2))}


def run_small_meetings(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                       calib_days: int = 14) -> dict[str, Any]:
    """Modèle spécialisé pour les petites courses contre la production, en walk-forward
    (evaluation/small_meetings.py ; critères fixés à l'avance)."""
    from sqlalchemy import text

    from horse_intelligence.evaluation import small_meetings as sm
    from horse_intelligence.evaluation.profile_search import prepare

    t0 = time.perf_counter()
    raw = load_raw(engine)
    ds = build_dataset(raw)
    if not audit_source_aggregates(ds.frame).passed:
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]
    with engine.connect() as conn:
        races = pd.read_sql(text("SELECT race_id, audience, prize_eur AS _prize FROM races"),
                            conn)
    frame = frame.merge(races, on="race_id", how="left")
    if "prize_eur" not in frame:
        frame["prize_eur"] = frame["_prize"]
    frame = frame.drop(columns="_prize")
    frame["small"] = sm.is_small(frame).astype(float)
    state: dict[str, Any] = {}

    def production(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        m = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL, with_market=True)
        m.fit(base, _with_market(calib))
        state.clear()
        state["prod"], state["key"] = m, id(base)
        sub = _with_market(test)
        return _masked(test, sub, m.predict(sub))

    def ctx_cal(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        if state.get("key") != id(base):
            raise RuntimeError("modèle de production du pli absent")
        prod = state["prod"]
        m = sm.ContextLgbm(LGBM_FEATURES, CATEGORICAL, with_market=True)
        m.booster, m.categories = prod.booster, prod.categories
        m.calibrate(_with_market(calib))
        sub = _with_market(test)
        return _masked(test, sub, m.predict(sub))

    def small_only(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        m = sm.ContextLgbm(LGBM_FEATURES, CATEGORICAL, with_market=True)
        m.fit_weighted(base[base["small"] == 1], _with_market(calib))
        sub = _with_market(test)
        return _masked(test, sub, m.predict(sub))

    def weighted(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        m = sm.ContextLgbm(LGBM_FEATURES, CATEGORICAL, with_market=True)
        w = np.where(base["small"].to_numpy() == 1, sm.SMALL_WEIGHT, 1.0)
        m.fit_weighted(base, _with_market(calib), w)
        sub = _with_market(test)
        return _masked(test, sub, m.predict(sub))

    runners: dict[str, ModelRunner] = {
        REFERENCE_MODEL: model_runners()[REFERENCE_MODEL], "lgbm_market": production,
        "ctx_cal": ctx_cal, "small_only": small_only, "weighted": weighted}
    wf = run_walk_forward(frame, runners, min_train_days, test_days, calib_days)
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = list(runners)
    mkt = _with_market(evald)
    mkt = mkt[mkt.groupby("race_id")[names].transform(lambda s: s.notna().all()).all(axis=1)]
    small = mkt[mkt["small"] == 1]
    big = mkt[mkt["small"] == 0]
    alpha_pred = 0.05 / len(sm.VARIANTS)
    alpha_bet = 0.05 / (2 * len(sm.VARIANTS))
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, pool, bet_type, combination, payout_per_eur FROM payouts "
            "WHERE pool = 'INTERNET' AND bet_type IN ('SIMPLE_GAGNANT', 'SIMPLE_PLACE')"),
            conn)
    f = prepare(small, payouts, "lgbm_market")
    f = f[f["profit_gagnant"].notna() | f["profit_placé"].notna()]
    bets = sm.betting(f, ["lgbm_market", *sm.VARIANTS], alpha_bet)
    success = any(bets[v][b]["all"].get("roi_ci", (0, 0))[0] > 0
                  for v in sm.VARIANTS for b in ("gagnant", "placé"))
    dates = sorted(mkt["race_date"].unique())
    return {
        "period": [str(dates[0]), str(dates[-1])] if dates else ["", ""],
        "n_races": int(mkt["race_id"].nunique()), "n_small": int(small["race_id"].nunique()),
        "n_folds": len(wf.folds), "seconds": round(time.perf_counter() - t0, 1),
        "errors": {k: v[:3] for k, v in wf.errors.items()},
        "metrics_small": {n: compute_metrics(small, small[n]).as_dict() for n in names},
        "small_vs_market": {n: paired_bootstrap(small, small[n], small[REFERENCE_MODEL])
                            for n in names if n != REFERENCE_MODEL},
        "big_vs_market": {n: paired_bootstrap(big, big[n], big[REFERENCE_MODEL])
                          for n in names if n != REFERENCE_MODEL},
        "variants_vs_prod": {v: _corrected_paired(small, v, "lgbm_market", alpha_pred)
                             for v in sm.VARIANTS},
        "variants_vs_prod_big": {v: paired_bootstrap(big, big[v], big["lgbm_market"])
                                 for v in sm.VARIANTS},
        "alpha_pred": alpha_pred, "alpha_bet": alpha_bet,
        "baseline": {b: gs_eval_all(f, b) for b in ("gagnant", "placé")},
        "betting": bets, "success": success,
    }


def gs_eval_all(f: pd.DataFrame, bet: str) -> dict[str, Any]:
    """Repère : tous les partants des petites courses, au même pari."""
    from horse_intelligence.evaluation import grand_search as gs

    col = f"profit_{bet}"
    sub = f.loc[f[col].notna(), ["race_id", "race_date", col]]
    return gs.evaluate(sub.groupby(["race_id", "race_date"], as_index=False).agg(
        stake=(col, "size"), profit=(col, "sum")))
