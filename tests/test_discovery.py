"""Laboratoire : un effet réel injecté est détecté et validé ; le bruit ne l'est pas."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from horse_intelligence.intelligence.anomaly_detection import fit_scores
from horse_intelligence.intelligence.bayesian_analysis import (
    actor_table,
    dispersion_message,
    fit_prior,
    persistence_test,
)
from horse_intelligence.intelligence.pattern_discovery import (
    STATUS_VALIDATED,
    ae_interval,
    benjamini_hochberg,
    discover,
)
from horse_intelligence.monitoring.drift_detection import feature_drift, psi


def make_frame(n_races: int, effect: float, seed: int, start: date = date(2026, 1, 1)
               ) -> pd.DataFrame:
    """Le marché ignore `rider_change` ; la vraie force est multipliée par `effect`."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = 10
        skill = rng.normal(0, 1, n)
        change = rng.integers(0, 2, n).astype(float)
        mkt = np.exp(skill) / np.exp(skill).sum()
        true = np.exp(skill + np.log(effect) * change)
        true /= true.sum()
        w = rng.choice(n, p=true)
        for i in range(n):
            rows.append({
                "race_id": r, "race_date": start + timedelta(days=r // 15), "num_pmu": i + 1,
                "won": float(i == w), "mkt_prob": mkt[i], "rider_change": change[i],
                "discipline": "trot_attele", "distance_m": 2700.0, "n_runners": float(n),
                "mkt_rank": float(np.argsort(np.argsort(-mkt))[i] + 1),
                "mus_last_pos": float(rng.integers(1, 11)), "mus_trend": rng.normal(0, 2),
                "draw_rel": (i + 1) / n, "age": float(rng.integers(3, 11)), "shoes": None,
                "horse_days_since_db": np.nan, "jockey_id": float(rng.integers(0, 60)),
                "skill": skill[i],
            })
    return pd.DataFrame(rows)


def test_benjamini_hochberg() -> None:
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    # p triées : 0.01→0.04, 0.03→0.06, 0.04→0.0533 ; monotonie : 0.03 prend 0.0533
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)


def test_ae_interval_contains_estimate() -> None:
    lo, hi = ae_interval(20, 10)
    assert lo < 2.0 < hi


def test_injected_effect_is_validated() -> None:
    disc = make_frame(1500, effect=1.6, seed=1)
    conf = make_frame(1000, effect=1.6, seed=2, start=date(2026, 6, 1))
    table, n_tests = discover(disc, conf, max_order=1)
    assert n_tests > 10
    hit = table[table["label"] == "changement de monte"]
    assert not hit.empty
    assert hit["status"].iloc[0] == STATUS_VALIDATED
    assert hit["ae"].iloc[0] > 1.15


def test_noise_produces_no_validated_pattern() -> None:
    disc = make_frame(1500, effect=1.0, seed=3)
    conf = make_frame(1000, effect=1.0, seed=4, start=date(2026, 6, 1))
    table, _ = discover(disc, conf, max_order=2)
    assert table.empty or (table["status"] != STATUS_VALIDATED).all()


def test_gamma_poisson_dispersion_test_is_calibrated() -> None:
    rng = np.random.default_rng(0)
    E = rng.uniform(5, 30, 300)
    # sous H0 (aucune dispersion) : environ 5 % de faux positifs, pas davantage
    rejections = [fit_prior(rng.poisson(E), E).dispersion_p < 0.05 for _ in range(200)]
    assert np.mean(rejections) < 0.10
    flat = fit_prior(rng.poisson(E), E)
    assert flat.dispersion_p > 0.05 or flat.a > 50  # jamais une forte dispersion
    lam = rng.gamma(5.0, 1 / 5.0, 300)  # écart-type relatif ≈ 45 %
    disp = fit_prior(rng.poisson(lam * E), E)
    assert disp.dispersion_p < 0.001 and 2.0 < disp.a < 15.0
    assert "Dispersion détectée" in dispersion_message(disp)


def test_actor_table_shrinks_small_samples() -> None:
    f = make_frame(600, effect=1.0, seed=5)
    tab, _ = actor_table(f, "jockey_id")
    raw_spread = tab["ae_raw"].std()
    assert tab["ae_post"].std() < raw_spread
    assert ((tab["ae_low"] <= tab["ae_post"]) & (tab["ae_post"] <= tab["ae_high"])).all()


def test_persistence_detects_true_skill() -> None:
    def frame(seed: int) -> pd.DataFrame:
        f = make_frame(1500, effect=1.0, seed=seed)
        rng = np.random.default_rng(seed + 100)
        # jockeys réellement meilleurs que ne le pensent les cotes
        boost = np.where(np.arange(60) < 20, 2.0, 1.0)
        f["won"] = 0.0
        for _, g in f.groupby("race_id"):
            w = g["mkt_prob"] * boost[g["jockey_id"].astype(int)]
            f.loc[rng.choice(g.index, p=w / w.sum()), "won"] = 1.0
        return f
    res = persistence_test(frame(10), frame(11), "jockey_id")
    assert res.p_value < 0.05 and res.spearman > 0


def test_anomaly_scores_flag_outliers() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    test = pd.DataFrame({"a": [0.0, 8.0], "b": [0.0, -8.0]})
    s = fit_scores(train, test, ["a", "b"])
    assert s[1] > s[0]


def test_psi_and_structural_drift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(size=2000), "horse_db_runs": rng.poisson(1, 2000)})
    same = pd.DataFrame({"x": rng.normal(size=2000), "horse_db_runs": rng.poisson(6, 2000)})
    assert psi(ref["x"], same["x"]) < 0.05
    shifted = same.assign(x=same["x"] + 1.5)
    t = feature_drift(ref, shifted, ["x", "horse_db_runs"]).set_index("feature")
    assert t.loc["x", "niveau"] == "dérive forte"
    assert t.loc["horse_db_runs", "niveau"].startswith("structurelle")


@pytest.mark.parametrize("n", [0])
def test_discover_handles_empty(n: int) -> None:
    empty = make_frame(5, 1.0, 0).iloc[n:0]
    table, tests = discover(empty, empty)
    assert table.empty and tests == 0
