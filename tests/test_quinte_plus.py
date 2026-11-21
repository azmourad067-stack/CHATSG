"""Quinté+ : règlement par rang (ordre, désordre, bonus 4sur5, bonus 3) et formules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import quinte_plus as qp


def _paid() -> dict[int, dict[str, object]]:
    """Arrivée 5-4-6-13-9 du 20/11/2026 (R1C4), rapports Internet par euro."""
    combos = ["5-4-6-13-9", "5-4-6-13-9", "5-4-6-13", "5-4-6-9", "5-4-13-9", "5-6-13-9",
              "4-6-13-9", "5-4-6", "1-2-3-4-5"]
    bets = ["QUINTE_PLUS", "QUINTE_PLUS_DESORDRE", *["QUINTE_PLUS"] * 6, "QUINTE_PLUS"]
    vals = [1175.5, 13.8, 1.4, 1.4, 1.4, 1.4, 1.4, 1.3, 9.0]
    payouts = pd.DataFrame({"race_id": [1] * 8 + [2], "pool": "INTERNET", "bet_type": bets,
                            "combination": combos, "payout_per_eur": vals})
    return qp.quinte_results(payouts)


def test_results_need_ordre_and_desordre() -> None:
    paid = _paid()
    assert set(paid) == {1}  # course 2 : pas de désordre collecté
    assert len(paid[1]["bonus4"]) == 5 and len(paid[1]["bonus3"]) == 1


def test_each_ticket_pays_its_best_rank_only() -> None:
    race = _paid()[1]
    assert qp.ticket_rank((5, 4, 6, 13, 9), race) == ("ordre", 1175.5)
    assert qp.ticket_rank((4, 5, 6, 13, 9), race) == ("desordre", 13.8)
    assert qp.ticket_rank((5, 4, 6, 13, 1), race) == ("bonus4", 1.4)  # 4 des 5 premiers
    assert qp.ticket_rank((4, 6, 13, 9, 1), race) == ("bonus4", 1.4)  # sans le gagnant
    assert qp.ticket_rank((5, 4, 6, 1, 2), race) == ("bonus3", 1.3)
    assert qp.ticket_rank((5, 4, 1, 2, 3), race) == (None, 0.0)


def test_formulas_ticket_counts_and_order() -> None:
    nums = np.arange(1, 13)
    score = np.linspace(1, 0.1, 12)
    for n in (5, 6, 7, 10):
        t = qp.combined_tickets(nums, score, n)
        assert len(t) == qp.n_tickets("combine", n) and len(set(t)) == len(t)
        assert all(list(c) == sorted(c) for c in t)  # écrites dans l'ordre du classement
    t = qp.one_base_tickets(nums, score, 6)
    assert len(t) == qp.n_tickets("base", 6) == 15 and all(c[0] == 1 for c in t)
    assert {h for c in t for h in c} == set(range(1, 8))


def test_simulate_combined_six() -> None:
    pred = pd.DataFrame({"race_id": [1] * 10, "num_pmu": [5, 4, 6, 13, 9, 1, 2, 3, 7, 8],
                         "race_date": [pd.Timestamp("2026-11-20").date()] * 10,
                         "s": np.linspace(1, 0.1, 10)})
    res = qp.simulate(pred, _paid(), lambda g: qp.combined_tickets(
        g["num_pmu"].to_numpy(), g["s"].to_numpy(), 6))
    row = res.iloc[0]
    # 6 combinaisons : 1 ordre exact + 5 avec 4 des 5 premiers (bonus 4sur5)
    assert row["stake"] == 12 and row["n_ordre"] == 1 and row["n_bonus4"] == 5
    assert round(row["ret"], 1) == round(2 * (1175.5 + 5 * 1.4), 1)
    assert qp.rank_rates(res)["ordre"] == 1.0


def test_quinte_plus_experiment_runs_without_quinte(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_quinte_plus
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_quinte_plus(repo.engine)
    assert rep["n_quinte"] == 0 and rep["success"] is False
