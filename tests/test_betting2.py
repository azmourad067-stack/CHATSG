"""Combinés au modèle placé, rapport final prévu, réglage de LightGBM."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from tests.synthetic import populate

from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.evaluation import payout_model as pm
from horse_intelligence.models.place_model import pl_place_column, pl_strengths_from_place
from horse_intelligence.models.ranking import place_probabilities


def test_pl_strengths_reproduce_place_probabilities() -> None:
    w = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05])
    s = pl_strengths_from_place(place_probabilities(w, 3), 3)
    np.testing.assert_allclose(s, w, atol=0.01)  # Harville inversé : on retrouve les forces
    q = np.array([0.8, 0.6, 0.45, 0.4, 0.3, 0.2, 0.15, 0.1])
    q = q * 3 / q.sum()
    np.testing.assert_allclose(place_probabilities(pl_strengths_from_place(q, 3), 3), q,
                               atol=0.02)


def test_pl_place_column_per_race() -> None:
    f = pd.DataFrame({"race_id": [1] * 8 + [2] * 5, "n_runners": [8.0] * 8 + [5.0] * 5,
                      "declared_runners": np.nan,
                      "q": [0.8, 0.6, 0.45, 0.4, 0.3, 0.2, 0.15, 0.1,
                            0.7, 0.5, 0.4, 0.25, 0.15]})
    s = pd.Series(pl_place_column(f, "q"))
    assert s.groupby(f["race_id"]).sum().round(6).tolist() == [1.0, 1.0]
    assert s[:8].is_monotonic_decreasing


def _payout_frame(n_days: int = 150) -> pd.DataFrame:
    """Courses de 8 partants où le rapport final vaut 0,8 × la cote de référence pour le
    favori du modèle (argent de dernière minute) et 1,0 × sinon."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_days):
        for r in range(4):
            rid = d * 10 + r
            p = rng.dirichlet(np.ones(8) * 2)
            model = p * np.exp(rng.normal(0, 0.2, 8))
            model /= model.sum()
            win = rng.choice(8, p=p)
            order = [win, *[i for i in np.argsort(-p) if i != win]][:3]
            for i in range(8):
                odds = 0.85 / p[i]
                fav = i == int(np.argmax(model))
                rows.append({
                    "race_id": rid, "race_date": date(2025, 1, 1) + timedelta(days=d),
                    "num_pmu": i + 1, "mkt_prob": p[i], "mkt_odds": odds,
                    "mkt_rank": int((p > p[i]).sum()) + 1, "mkt_pool": "INTERNET",
                    "n_runners": 8.0, "declared_runners": 8.0, "discipline": "trot_attele",
                    "won": float(i == win), "finish_position": order.index(i) + 1
                    if i in order else 5.0, "lgbm_market": model[i],
                    "place_market": min(0.95, 3 * model[i]),
                    "payout_win": (0.8 if fav else 1.0) * odds if i == win else np.nan,
                    "payout_place": 1.5 if i in order else np.nan,
                    "has_win_payouts": True, "has_place_payouts": True})
    return pd.DataFrame(rows)


def test_payout_model_learns_late_money_without_leakage() -> None:
    f = pm.predict_payouts(_payout_frame(), "lgbm_market", "place_market",
                           min_train_months=2, min_train_rows=100)
    month = pd.to_datetime(f["race_date"]).dt.to_period("M")
    first_two = month < sorted(month.unique())[2]
    assert f.loc[first_two, "exp_payout_win"].isna().all()  # rien sans historique
    late = f[~first_two]
    fav = late.groupby("race_id")["lgbm_market"].transform("max") == late["lgbm_market"]
    ratio = late["exp_payout_win"] / late["mkt_odds"]
    assert ratio[fav].median() < ratio[~fav].median()  # baisse de dernière minute apprise
    acc = pm.payout_accuracy(f)
    assert acc["n_winners"] > 0 and acc["diff"] < 0  # mieux que la cote de référence
    assert late["exp_payout_place"].notna().all()
    out = pm.select_then_test(f, "win", "predicted", "lgbm_market", "place_market")
    assert out["n_rules_tried"] == len(pm.rule_grid("win", "predicted"))


def test_ev_rules_one_bet_per_race() -> None:
    f = pd.DataFrame({"race_id": [1, 1, 2, 2], "num_pmu": [1, 2, 1, 2],
                      "race_date": [date(2025, 1, 1)] * 2 + [date(2025, 1, 2)] * 2,
                      "p": [0.5, 0.3, 0.2, 0.1], "mkt_odds": [2.0, 5.0, 4.0, 20.0],
                      "exp_payout_win": [1.8, 4.0, 3.0, 9.0], "payout_win": [np.nan, 4.2,
                                                                             np.nan, np.nan],
                      "has_win_payouts": True})
    ref = pm.rule_bets(f, pm.EvRule("win", "reference", 0.0, 0.0), "p", "p")
    assert ref["num_pmu"].tolist() == [2, 2]  # EV 0,5 et 1,0 sur la cote de référence
    pred = pm.rule_bets(f, pm.EvRule("win", "predicted", 0.0, 0.0), "p", "p")
    # au rapport prévu, seul le n° 2 de la course 1 reste à EV ≥ 0 (0,3 × 4 − 1 = 0,2)
    assert pred["race_id"].tolist() == [1] and pred["profit"].round(2).tolist() == [3.2]


def test_betting2_experiment_end_to_end(repo: Repository) -> None:
    from horse_intelligence.services.experiment import run_betting2_experiment
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_betting2_experiment(repo.engine)
    assert rep["n_races"] > 0 and isinstance(rep["success"], bool)
    assert set(rep["combos"]) == {"lgbm_market", "pl_place"}
    assert set(rep["singles"]) == {"gagnant (cote de référence)", "gagnant (rapport prévu)",
                                   "placé (rapport prévu)"}


def test_tuning_experiment_end_to_end(repo: Repository) -> None:
    from horse_intelligence.services.experiment import run_tuning_experiment

    populate(repo, days=30)
    rep = run_tuning_experiment(repo.engine, min_train_days=12, test_days=4, calib_days=4,
                                configs={"actuel": ({}, 50),
                                         "petit": ({"num_leaves": 7}, 50),
                                         "lent": ({"learning_rate": 0.01}, 80)})
    assert rep["chosen"] in ("petit", "lent")
    assert set(rep["second_half"]) == {"petit", "lent"}
    assert isinstance(rep["success"], bool) and not rep["errors"]
    assert isinstance(rep["success_both_halves"], bool)
    assert rep["selection_period"][1] < rep["test_period"][0]


@pytest.mark.parametrize("bet", ["win", "place"])
def test_rule_grid_sizes(bet: str) -> None:
    assert len(pm.rule_grid(bet, "predicted")) == (36 if bet == "win" else 20)


def test_race_model_uses_tuned_settings_place_model_does_not() -> None:
    from horse_intelligence.models.lightgbm_model import LgbmRaceModel
    from horse_intelligence.models.place_model import PlaceModel

    race = LgbmRaceModel(["x"], [])
    assert race.params["num_leaves"] == 31 and race.params["min_data_in_leaf"] == 60
    assert race.num_rounds == 400
    place = PlaceModel(["x"], [])
    assert place.params["num_leaves"] == 15 and place.num_rounds == 300


def test_large_tuning_baseline_is_production() -> None:
    from horse_intelligence.models.lightgbm_model import (
        DEFAULT_PARAMS,
        RACE_NUM_ROUNDS,
        RACE_PARAMS,
    )
    from horse_intelligence.services.experiment import TUNING_LARGE_CONFIGS

    params, rounds = TUNING_LARGE_CONFIGS["production"]
    assert {**DEFAULT_PARAMS, **params} == RACE_PARAMS and rounds == RACE_NUM_ROUNDS
