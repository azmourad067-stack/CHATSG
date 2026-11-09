"""Simulation des paris combinés : probabilités, règlement et protocole."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.combo_simulator import (
    ComboStrategy,
    combo_probabilities,
    select_then_test_combos,
    simulate_combos,
    winning_combos,
)


def test_combo_probabilities_are_consistent() -> None:
    rng = np.random.default_rng(0)
    nums, p = np.array([1, 2, 3, 4, 5]), np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    cg = combo_probabilities(nums, p, 2, 2, rng, 20_000)
    assert abs(sum(cg.values()) - 1.0) < 1e-9          # une seule paire gagnante par arrivée
    assert abs(sum(combo_probabilities(nums, p, 2, 3, rng, 20_000).values()) - 3.0) < 1e-9
    # Harville exact pour la paire des deux favoris
    exact = 0.4 * 0.3 / 0.6 + 0.3 * 0.4 / 0.7
    assert abs(cg[(1, 2)] - exact) < 0.01


def test_winning_combos_parsing_ignores_non_runner_rows() -> None:
    pay = pd.DataFrame({"race_id": [1, 1, 1, 1], "pool": "INTERNET",
                        "bet_type": ["COUPLE_GAGNANT", "TRIO", "TRIO", "COUPLE_PLACE"],
                        "combination": ["4-1", "4-1-7", "9-5-NP", "8-6"],
                        "payout_per_eur": [12.0, 30.0, 5.0, 4.0]})
    w = winning_combos(pay)
    assert w[(1, "COUPLE_GAGNANT")] == {(1, 4): 12.0}
    assert w[(1, "TRIO")] == {(1, 4, 7): 30.0}
    assert w[(1, "COUPLE_PLACE")] == {(6, 8): 4.0}


def _cand() -> pd.DataFrame:
    rows = []
    for d in range(80):
        day = date(2026, 1, 1) + timedelta(days=d)
        for c, pm, pk, pay in (((1, 2), 0.30, 0.30, 3.0 if d % 3 == 0 else 0.0),
                               ((3, 4), 0.10, 0.05, 15.0 if d % 7 == 0 else 0.0)):
            rows.append({"race_id": d, "race_date": day, "bet_type": "COUPLE_GAGNANT",
                         "combo": c, "p_model": pm, "p_market": pk, "payout": pay})
    return pd.DataFrame(rows)


def test_simulation_settles_one_combo_per_race() -> None:
    cand = _cand()
    fav = simulate_combos(cand, ComboStrategy("fav", "COUPLE_GAGNANT", "market"))
    assert fav.n_bets == 80 and fav.n_races == 80
    value = simulate_combos(cand, ComboStrategy("v", "COUPLE_GAGNANT", "ratio", 1.5, 0.05))
    assert value.n_bets == 80                           # seule (3, 4) a un ratio de 2
    assert abs(value.roi - (15.0 * len(range(0, 80, 7)) / 80 - 1)) < 1e-9


def test_rule_is_chosen_on_selection_then_tested_later() -> None:
    sel = select_then_test_combos(_cand(), "COUPLE_GAGNANT")
    assert sel.selection_period[1] < sel.test_period[0]
    assert sel.chosen_on_selection is not None


def test_combo_backtest_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    """Entraînement synthétique, rapports de couplé et trio ajoutés, simulation complète."""
    from sqlalchemy import select
    from tests.synthetic import populate

    from horse_intelligence.data.schemas.domain import PayoutRecord
    from horse_intelligence.data.storage import database as db
    from horse_intelligence.services.experiment import run_combo_backtest
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=30)
    train_and_evaluate(repo.engine, min_train_days=14, test_days=5, calib_days=4,
                       register=False, n_members=2)
    with repo.engine.connect() as conn:
        res = pd.DataFrame(conn.execute(select(db.race_results.c.race_id,
                                               db.race_results.c.num_pmu,
                                               db.race_results.c.finish_position)).all())
    for race_id, g in res.dropna().sort_values("finish_position").groupby("race_id"):
        top = [str(int(n)) for n in g["num_pmu"].head(3)]
        repo.save_payouts(int(race_id), [
            PayoutRecord("COUPLE_GAGNANT", "-".join(top[:2]), 12.0, "INTERNET"),
            PayoutRecord("TRIO", "-".join(top), 40.0, "INTERNET")], datetime.now(UTC))
    rep = run_combo_backtest(repo.engine)
    assert set(rep["bets"]) == {"COUPLE_GAGNANT", "TRIO"}
    fav = rep["bets"]["COUPLE_GAGNANT"]["all_period"][0]
    assert fav["n_bets"] > 0 and 0 <= fav["hit_rate"] <= 1
