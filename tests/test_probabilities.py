from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horse_intelligence.evaluation.metrics import auc_score, compute_metrics, paired_bootstrap
from horse_intelligence.models.baseline import CalibratedMarket
from horse_intelligence.models.ranking import (
    ConditionalLogit,
    group_codes,
    group_softmax,
    paid_places,
    place_probabilities,
)


def _synthetic(n_races: int = 400, n: int = 8, beta: float = 1.2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        x = rng.normal(size=n)
        p = np.exp(beta * x) / np.exp(beta * x).sum()
        w = rng.choice(n, p=p)
        for i in range(n):
            rows.append({"race_id": r, "x": x[i], "won": float(i == w), "true_p": p[i],
                         "race_date": r // 20, "num_pmu": i + 1, "finish_position": np.nan,
                         "n_runners": n})
    return pd.DataFrame(rows)


def test_group_softmax_sums_to_one() -> None:
    codes, n = group_codes([1, 1, 1, 2, 2])
    p = group_softmax([1000.0, 999.0, -5.0, 0.0, 0.0], codes, n)  # stabilité numérique
    sums = np.bincount(codes, weights=p)
    assert np.allclose(sums, 1.0) and np.all(np.isfinite(p))
    assert p[3] == pytest.approx(0.5)


def test_conditional_logit_recovers_coefficient() -> None:
    df = _synthetic()
    m = ConditionalLogit(l2=1e-6).fit(df[["x"]], df["race_id"], df["won"])
    beta_hat = m.coef_[0] / m.scale_[0]
    assert beta_hat == pytest.approx(1.2, abs=0.25)
    p = m.predict_proba(df[["x"]], df["race_id"])
    assert np.allclose(df.assign(p=p).groupby("race_id")["p"].sum(), 1.0)


def test_conditional_logit_ignores_multi_winner_races_and_nan() -> None:
    df = _synthetic(50)
    df.loc[df.index[:2], "won"] = 1.0  # course 0 : deux gagnants → exclue
    df.loc[df.index[5], "x"] = np.nan
    m = ConditionalLogit().fit(df[["x"]], df["race_id"], df["won"])
    assert np.all(np.isfinite(m.predict_proba(df[["x"]], df["race_id"])))


def test_logit_serialization_roundtrip() -> None:
    df = _synthetic(50)
    m = ConditionalLogit().fit(df[["x"]], df["race_id"], df["won"], ["x"])
    m2 = ConditionalLogit.from_dict(m.to_dict())
    assert np.allclose(m.scores(df[["x"]]), m2.scores(df[["x"]]))


def test_place_probabilities() -> None:
    p = np.array([0.5, 0.3, 0.15, 0.05])
    pl = place_probabilities(p, 2)
    assert pl.sum() == pytest.approx(2.0)
    # Harville exact pour le n°1 : 0,5 + Σ_j p_j * 0,5 / (1 - p_j)
    expected = 0.5 + sum(p[j] * 0.5 / (1 - p[j]) for j in (1, 2, 3))
    assert pl[0] == pytest.approx(expected)
    assert np.all(np.diff(pl) <= 0)  # ordre préservé
    assert place_probabilities(p, 0).sum() == 0
    assert np.allclose(place_probabilities(p, 4), 1.0)
    discounted = place_probabilities(p, 2, discounts=(1.0, 0.8))
    assert discounted.sum() == pytest.approx(2.0)


@pytest.mark.parametrize(("n", "places"), [(3, 0), (4, 2), (7, 2), (8, 3), (16, 3)])
def test_paid_places(n: int, places: int) -> None:
    assert paid_places(n) == places


def test_calibrated_market_corrects_power() -> None:
    df = _synthetic(600, seed=1)
    # marché biaisé : probabilités « aplaties » (q ∝ p^0.6)
    q = df["true_p"] ** 0.6
    df["mkt_prob"] = q / q.groupby(df["race_id"]).transform("sum")
    df["mkt_log_prob"] = np.log(df["mkt_prob"])
    cm = CalibratedMarket().fit(df)
    alpha = cm.model.coef_[0] / cm.model.scale_[0]
    assert alpha == pytest.approx(1 / 0.6, rel=0.25)
    assert compute_metrics(df, cm.predict(df)).log_loss < compute_metrics(
        df, df["mkt_prob"]).log_loss


def test_auc_and_bootstrap() -> None:
    assert auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    df = _synthetic(300, seed=2)
    unif = np.full(len(df), 1 / 8)
    res = paired_bootstrap(df, df["true_p"], unif)
    assert res["ci_high"] < 0  # les vraies probabilités battent l'uniforme
    m = compute_metrics(df, df["true_p"])
    assert m.n_races == 300 and 0 < m.ece < 0.05


@pytest.mark.parametrize(("n", "k", "disc"), [(5, 2, (1.0, 1.0)), (9, 3, (1.0, 0.8, 0.7)),
                                              (16, 3, (1.0, 1.0, 1.0))])
def test_vectorized_places_match_enumeration(n: int, k: int, disc: tuple[float, ...]) -> None:
    from itertools import permutations

    p = np.random.default_rng(n).dirichlet(np.ones(n))
    strengths = [p ** disc[min(i, len(disc) - 1)] for i in range(k)]
    ref = np.zeros(n)
    for order in permutations(range(n), k):
        prob, used = 1.0, []
        for j, i in enumerate(order):
            s = strengths[j]
            prob *= s[i] / (s.sum() - s[used].sum())
            used.append(i)
        for i in order:
            ref[i] += prob
    assert np.allclose(place_probabilities(p, k, disc), ref, atol=1e-12)
