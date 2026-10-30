from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from horse_intelligence.evaluation.betting_simulator import (
    Strategy,
    reference_strategies,
    select_bets,
    select_then_test,
    simulate,
)


def _pred() -> pd.DataFrame:
    # 3 courses de 3 partants ; le n°1 de chaque course a l'espérance la plus haute
    rows = []
    for r in range(3):
        for n, (p, odds) in enumerate([(0.5, 3.0), (0.3, 2.5), (0.2, 6.0)], start=1):
            rows.append({"race_id": r, "race_date": date(2026, 1, 1) + timedelta(days=r),
                         "num_pmu": n, "model": p, "mkt_prob": 1 / odds, "mkt_odds": odds,
                         "declared_runners": 3, "won": float(n == 1 and r != 1)})
    return pd.DataFrame(rows)


def _payouts() -> pd.DataFrame:
    return pd.DataFrame([
        {"race_id": 0, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT", "combination": "1",
         "payout_per_eur": 2.8},
        {"race_id": 1, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT", "combination": "2",
         "payout_per_eur": 2.4},
        # course 2 : aucun rapport collecté → exclue
    ])


def test_flat_simulation_uses_official_payouts() -> None:
    res = simulate(_pred(), _payouts(), Strategy("t", prob_col="model", min_ev=0.2))
    # course 0 : n°1 gagne, rapport 2,80 ; course 1 : n°1 perd ; course 2 exclue
    assert res.n_bets == 2 and res.n_races_without_payouts == 1
    assert res.returned == pytest.approx(2.8)
    assert res.roi == pytest.approx((2.8 - 2) / 2)
    assert res.hit_rate == 0.5 and res.longest_losing_streak == 1
    assert res.max_drawdown == pytest.approx(1.0)  # +1,80 puis −1 € depuis le sommet


def test_selection_respects_thresholds_and_one_bet_per_race() -> None:
    bets = select_bets(_pred(), Strategy("t", prob_col="model", min_ev=0.0))
    assert bets.groupby("race_id").size().max() == 1
    assert (bets["num_pmu"] == 1).all()  # meilleure espérance (0,5 × 3 − 1 = 0,5)
    none = select_bets(_pred(), Strategy("t", prob_col="model", min_ev=1.0))
    assert none.empty
    capped = select_bets(_pred(), Strategy("t", prob_col="model", min_ev=-1, max_odds=2.6))
    assert (capped["mkt_odds"] <= 2.6).all()


def test_favourite_reference_picks_highest_probability() -> None:
    fav = reference_strategies()[0]
    bets = select_bets(_pred(), fav)
    assert (bets["num_pmu"] == 2).all()  # cote 2,5 = favori du marché


def test_kelly_stakes_are_bounded() -> None:
    s = Strategy("k", prob_col="model", min_ev=0.0, staking="kelly", kelly_fraction=0.5,
                 max_stake_share=0.05)
    res = simulate(_pred(), _payouts(), s, bankroll=100.0)
    assert res.equity and res.final_bankroll is not None
    for e in res.equity:  # mise ≤ 5 % de la bankroll disponible avant le pari
        before = e["bankroll"] - (e["return"] - e["stake"])
        assert 0 < e["stake"] <= 0.05 * before + 0.01


def test_place_bets_use_place_payouts() -> None:
    pay = pd.DataFrame([{"race_id": 0, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT",
                         "combination": "1", "payout_per_eur": 2.8},
                        {"race_id": 0, "pool": "INTERNET", "bet_type": "SIMPLE_PLACE",
                         "combination": "1", "payout_per_eur": 1.3}])
    pred = _pred()
    pred["declared_runners"] = 5  # 2 places payées
    res = simulate(pred[pred["race_id"] == 0], pay,
                   Strategy("p", "SIMPLE_PLACE", "model", min_prob=0.5))
    assert res.n_bets == 1 and res.returned == pytest.approx(1.3)


def test_select_then_test_never_tunes_on_test() -> None:
    rng = np.random.default_rng(0)
    rows, pays = [], []
    for r in range(400):
        p = rng.dirichlet(np.ones(8))
        w = rng.choice(8, p=p)
        for i in range(8):
            rows.append({"race_id": r, "race_date": date(2026, 1, 1) + timedelta(days=r // 10),
                         "num_pmu": i + 1, "model": p[i], "mkt_prob": p[i],
                         "mkt_odds": 0.85 / p[i], "declared_runners": 8,
                         "won": float(i == w)})
        pays.append({"race_id": r, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT",
                     "combination": str(w + 1), "payout_per_eur": 0.85 / p[w]})
    out = select_then_test(pd.DataFrame(rows), pd.DataFrame(pays), "model")
    assert out.selection_period[1] < out.test_period[0]
    # le marché est exact et prélève 15 % : aucune règle ne doit être annoncée rentable
    assert out.test is None or out.test["roi"] < 0.15
    assert out.references_test


def test_ties_are_broken_deterministically() -> None:
    pred = _pred()
    pred["model"] = 1 / 3  # tous à égalité
    pred["mkt_odds"] = 3.0
    a = select_bets(pred, Strategy("t", prob_col="model", min_ev=-1))
    b = select_bets(pred.sample(frac=1, random_state=1), Strategy("t", prob_col="model",
                                                                  min_ev=-1))
    assert (a["num_pmu"] == 1).all() and list(a["num_pmu"]) == list(b["num_pmu"])
