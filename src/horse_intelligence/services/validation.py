"""Phase 5 : validation complète — rentabilité théorique, robustesse, audit des fuites."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from horse_intelligence.evaluation.betting_simulator import (
    reference_strategies,
    select_then_test,
    simulate,
)
from horse_intelligence.evaluation.leakage_checks import run_leakage_checks
from horse_intelligence.evaluation.metrics import compute_metrics
from horse_intelligence.evaluation.robustness import mask_features, segment_table, stability_summary
from horse_intelligence.intelligence.feature_engineering import build_dataset, load_raw
from horse_intelligence.models.lightgbm_model import LgbmRaceModel
from horse_intelligence.services.training import (
    CATEGORICAL,
    LGBM_FEATURES,
    NUMERIC_FEATURES,
    REFERENCE_MODEL,
    load_oos_predictions,
)

MODELS = ("lgbm_market", "lgbm_form", REFERENCE_MODEL)
MASK_SHARE = 0.2
SEEDS = (1, 2, 3)


def _with_market(f: pd.DataFrame) -> pd.DataFrame:
    return f[f.groupby("race_id")["mkt_prob"].transform(lambda s: s.notna().all())]


def _robustness(raw: dict[str, pd.DataFrame], test_days: int = 7,
                calib_days: int = 7) -> dict[str, Any]:
    """Dernier pli : sensibilité aux données manquantes et à l'aléa d'apprentissage."""
    frame = build_dataset(raw).frame
    frame = frame[frame["won"].notna()]
    dates = sorted(frame["race_date"].unique())
    test_d, calib_d = dates[-test_days:], dates[-test_days - calib_days:-test_days]
    base = frame[frame["race_date"] < calib_d[0]]
    calib = _with_market(frame[frame["race_date"].isin(calib_d)])
    test = _with_market(frame[frame["race_date"].isin(test_d)])

    def fit(seed: int) -> LgbmRaceModel:
        m = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL, with_market=True)
        m.params = {**m.params, "seed": seed}
        return m.fit(base, calib)

    models = [fit(s) for s in SEEDS]
    full = [compute_metrics(test, m.predict(test)).log_loss for m in models]
    masked_test = mask_features(test, NUMERIC_FEATURES, MASK_SHARE)
    p_masked = models[0].predict(masked_test)
    sums = pd.Series(p_masked, index=test.index).groupby(test["race_id"]).sum()
    masked = compute_metrics(test, p_masked).log_loss
    market = compute_metrics(test, test["mkt_prob"]).log_loss
    return {
        "test_period": [str(test_d[0]), str(test_d[-1])], "races": int(test["race_id"].nunique()),
        "seed_log_losses": full, "seed_spread": float(np.max(full) - np.min(full)),
        "masked_share": MASK_SHARE, "log_loss_masked": masked,
        "degradation_masked": masked - full[0], "market_log_loss": market,
        "probabilities_valid": bool(np.isfinite(p_masked).all()
                                    and np.allclose(sums.to_numpy(), 1.0)),
    }


def run_validation(engine: Engine, reports_dir: Path) -> dict[str, Any]:
    pred = load_oos_predictions(reports_dir)
    if pred is None:
        raise ValueError("aucune prédiction hors échantillon : lancez scripts/train_models.py")
    with engine.connect() as conn:
        payouts = pd.read_sql(text("SELECT race_id, pool, bet_type, combination, "
                                   "payout_per_eur FROM payouts"), conn)
    raw = load_raw(engine)
    models = [m for m in MODELS if m in pred.columns]
    mkt = _with_market(pred)

    betting = {m: asdict(select_then_test(mkt, payouts, m)) for m in models}
    refs_all = [simulate(mkt, payouts, s).as_dict() for s in reference_strategies()]
    segments = segment_table(mkt, "lgbm_market", REFERENCE_MODEL) \
        if "lgbm_market" in models else []
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "period": [str(pred["race_date"].min()), str(pred["race_date"].max())],
        "races": int(mkt["race_id"].nunique()),
        "races_with_payouts": int(mkt["race_id"].drop_duplicates().isin(
            payouts.loc[(payouts["pool"] == "INTERNET")
                        & (payouts["bet_type"] == "SIMPLE_GAGNANT"), "race_id"]).sum()),
        "betting": betting, "references_all_period": refs_all,
        "segments": segments, "stability": stability_summary(segments),
        "robustness": _robustness(raw), "leakage": run_leakage_checks(raw),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    (reports_dir / f"validation-{datetime.now(UTC):%Y%m%d-%H%M%S}.json").write_text(body)
    (reports_dir / "validation-latest.json").write_text(body)
    return report


def latest_validation(reports_dir: Path) -> dict[str, Any] | None:
    path = reports_dir / "validation-latest.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data
