"""Tiercé du quinté : rapports ordre / désordre, combinaisons et règlement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.data.processors.normalization import payout_bet_type, payouts_from_pmu
from horse_intelligence.data.schemas.pmu import PmuRapportPari
from horse_intelligence.evaluation import tierce_quinte as tq


def test_desordre_line_is_kept_apart_from_ordre() -> None:
    assert payout_bet_type("E_TIERCE", "e-Tiercé Ordre") == "TIERCE"
    assert payout_bet_type("E_TIERCE", "e-Tiercé Désordre") == "TIERCE_DESORDRE"
    assert payout_bet_type("QUINTE_PLUS", "Quinté+ Desordre") == "QUINTE_PLUS_DESORDRE"
    assert payout_bet_type("E_SIMPLE_GAGNANT", None) == "SIMPLE_GAGNANT"
    raw = [PmuRapportPari.model_validate({"typePari": "E_TIERCE", "rapports": [
        {"libelle": "e-Tiercé Ordre", "combinaison": "5-4-6", "dividendePourUnEuro": 11260},
        {"libelle": "e-Tiercé Désordre", "combinaison": "5-4-6", "dividendePourUnEuro": 1960}]})]
    recs = payouts_from_pmu(raw, "INTERNET")
    assert [(r.bet_type, r.combination, r.payout_per_eur) for r in recs] == [
        ("TIERCE", "5-4-6", 112.6), ("TIERCE_DESORDRE", "5-4-6", 19.6)]


def _paid() -> dict[int, dict[str, object]]:
    payouts = pd.DataFrame({
        "race_id": [1, 1, 2], "pool": "INTERNET",
        "bet_type": ["TIERCE", "TIERCE_DESORDRE", "TIERCE"],
        "combination": ["5-4-6", "5-4-6", "1-2-3"], "payout_per_eur": [112.6, 19.6, 50.0]})
    return tq.tierce_results(payouts)


def test_tierce_results_need_both_lines() -> None:
    paid = _paid()
    assert set(paid) == {1}  # course 2 : pas de désordre collecté, exclue
    assert paid[1]["ordre"] == {(5, 4, 6): 112.6}


def test_ticket_return_ordre_desordre_miss() -> None:
    race = _paid()[1]
    assert tq.ticket_return([(5, 4, 6)], race) == (112.6, True)
    assert tq.ticket_return([(4, 5, 6)], race) == (19.6, False)
    assert tq.ticket_return([(1, 2, 3)], race) == (0.0, False)


def test_top6_gives_twenty_tickets_in_model_order() -> None:
    nums = np.arange(1, 11)
    score = np.array([0.05, 0.3, 0.1, 0.2, 0.02, 0.15, 0.08, 0.04, 0.03, 0.03])
    t = tq.top6_tickets(nums, score)
    assert len(t) == 20 and len(set(t)) == 20
    assert t[0] == (2, 4, 6)  # les trois meilleurs, dans l'ordre du score
    assert {n for c in t for n in c} == {2, 4, 6, 3, 7, 1}


def test_proba20_tickets_are_twenty_distinct_trios() -> None:
    nums = np.arange(1, 13)
    p = np.linspace(2, 0.5, 12)
    p /= p.sum()
    t = tq.proba20_tickets(nums, p, p, np.random.default_rng(0), n_samples=3000)
    assert len(t) == 20 and len({frozenset(c) for c in t}) == 20
    assert all(list(c) == sorted(c) for c in t)  # ici l'ordre du modèle = numéros croissants


def test_simulate_and_summary() -> None:
    pred = pd.DataFrame({"race_id": [1] * 8, "race_date": [pd.Timestamp("2025-01-01").date()] * 8,
                         "num_pmu": range(1, 9),
                         "s": [0.05, 0.1, 0.05, 0.2, 0.3, 0.15, 0.1, 0.05]})
    res = tq.simulate(pred, _paid(), lambda g: tq.top6_tickets(g["num_pmu"].to_numpy(),
                                                               g["s"].to_numpy()))
    # 6 premiers : 5, 4, 6, 2, 7, 1 → la combinaison (5, 4, 6) est écrite dans l'ordre exact
    assert res.loc[0, "stake"] == 20 and res.loc[0, "ret"] == 112.6 and res.loc[0, "exact"]
    s = tq.summarize(res)
    assert s["n_races"] == 1 and round(s["profit"], 1) == 92.6 and s["hit_rate"] == 1.0


def test_tierce_quinte_experiment_runs_without_quinte(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_tierce_quinte
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_tierce_quinte(repo.engine)
    assert rep["n_quinte"] == 0 and rep["success"] is False
    assert repo.quinte_races_missing_bet("TIERCE_DESORDRE").empty
