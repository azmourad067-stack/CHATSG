"""Simulation de bankroll : ajustement de rentabilité, séries, creux et ruine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horse_intelligence.evaluation.bankroll import (
    _longest_losing_streak,
    rescale_to_roi,
    run_scenarios,
    simulate,
)
from horse_intelligence.evaluation.frozen_rules import FROZEN_RULES, rule_bets


def test_rescale_keeps_losses_and_hits_target_roi() -> None:
    profits = np.array([-1.0, -1.0, -1.0, 5.0, -1.0, 2.0])
    out = rescale_to_roi(profits, -0.10)
    assert out.mean() == pytest.approx(-0.10)
    assert (out[profits == -1.0] == -1.0).all()  # les défaites restent des défaites
    with pytest.raises(ValueError):
        rescale_to_roi(np.array([-1.0, -1.0]), 0.0)


def test_longest_losing_streak() -> None:
    paths = np.array([[-1, -1, 2, -1, -1, -1, 0.5], [3, 3, 3, 3, 3, 3, 3]], dtype=float)
    assert _longest_losing_streak(paths).tolist() == [3, 0]


def test_simulate_known_cases() -> None:
    # toujours perdant : gain final = −n, série = n, creux = n, ruine certaine
    r = simulate(np.array([-1.0]), n_bets=40, n_sims=200, bankrolls=(10, 50))
    assert r.final_quantiles["médiane"] == -40
    assert r.longest_losing_streak["médiane"] == 40
    assert r.max_drawdown["P99"] == 40
    assert r.p_ruin == {10: 1.0, 50: 0.0}
    assert r.p_profit == 0.0
    # toujours gagnant : jamais de creux ni de ruine
    r = simulate(np.array([1.0]), n_bets=40, n_sims=200, bankrolls=(1,))
    assert r.p_profit == 1.0 and r.max_drawdown["P99"] == 0 and r.p_ruin == {1: 0.0}


def test_scenarios_order_by_roi() -> None:
    rng = np.random.default_rng(1)
    profits = np.where(rng.random(2000) < 0.15, rng.uniform(3, 12, 2000), -1.0)
    res = run_scenarios(profits, n_bets=400, n_sims=2000)
    by_name = {r.scenario: r for r in res}
    assert by_name["neutre"].roi == pytest.approx(0.0)
    assert by_name["pessimiste (rang 3 ordinaire)"].roi == pytest.approx(-0.13)
    # plus la rentabilité est basse, plus la ruine est probable
    ruin = [r.p_ruin[100] for r in res[1:]]
    assert ruin == sorted(ruin)


def test_rule_bets_matches_frozen_rule_selection() -> None:
    rule = next(r for r in FROZEN_RULES if r.value == "Trot G")
    frame = pd.DataFrame({
        "race_id": [1, 1, 1, 2, 2, 2], "num_pmu": [1, 2, 3, 1, 2, 3],
        "race_date": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-01"] * 3).date,
        "p": [0.5, 0.3, 0.2, 0.6, 0.3, 0.1], "n_runners": 3, "distance_m": 2100,
        "reunion_num": 1, "label": "PRIX X", "conditions_text": "Course G - Attelé",
        "discipline": "trot_attele"})
    payouts = pd.DataFrame({
        "race_id": [1, 2], "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT",
        "combination": ["3", "1"], "payout_per_eur": [7.5, 1.4]})
    bets = rule_bets(frame, payouts, "p", rule)
    # 3e du modèle : n°3 dans chaque course ; triés par date ; gagnant seulement en course 1
    assert bets["race_id"].tolist() == [2, 1]
    assert bets["num_pmu"].tolist() == [3, 3]
    assert bets["profit"].tolist() == [-1.0, 6.5]
