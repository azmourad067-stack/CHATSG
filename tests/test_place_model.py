"""Modèle placé : cible, ajustement par course, Harville, règles de simple placé."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.synthetic import populate

from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.evaluation.place_betting import (
    PlaceRule,
    place_payouts,
    rule_bets,
    rule_grid,
)
from horse_intelligence.models.place_model import (
    harville_place,
    normalize_to_places,
    place_log_losses,
    placed_target,
)
from horse_intelligence.models.ranking import place_probabilities


def _race(race_id: int, n: int, declared: float | None = None) -> pd.DataFrame:
    return pd.DataFrame({"race_id": race_id, "num_pmu": range(1, n + 1),
                         "finish_position": [*range(1, n), np.nan], "won": 0.0,
                         "n_runners": float(n), "declared_runners": declared})


def test_placed_target_follows_paid_places() -> None:
    f = pd.concat([_race(1, 9), _race(2, 6), _race(3, 3), _race(4, 7, declared=8.0)],
                  ignore_index=True)
    f.loc[f["race_id"] == 2, "won"] = np.nan  # pas de résultat
    y = placed_target(f)
    assert y[f["race_id"] == 1].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]  # non classé = 0
    assert y[f["race_id"] == 2].isna().all()
    assert y[f["race_id"] == 3].isna().all()  # moins de 4 partants : pas de placé
    # 8 déclarés dont un non-partant : 3 places payées
    assert y[f["race_id"] == 4].sum() == 3


def test_normalize_to_places_sums_to_k_and_keeps_order() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=14)
    races = np.repeat([1, 2], [9, 5])
    k = np.repeat([3.0, 2.0], [9, 5])
    p = normalize_to_places(z, races, k)
    assert p[:9].sum() == pytest.approx(3.0, abs=1e-6)
    assert p[9:].sum() == pytest.approx(2.0, abs=1e-6)
    assert (np.argsort(p[:9]) == np.argsort(z[:9])).all()
    assert ((p > 0) & (p < 1)).all()


def test_harville_place_matches_exact_formula() -> None:
    f = _race(1, 8)
    w = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03])
    np.testing.assert_allclose(harville_place(f, w), place_probabilities(w, 3))
    # une course parfaitement prédite a une log-loss quasi nulle
    perfect = placed_target(f).to_numpy()
    assert place_log_losses(f, perfect).iloc[0] < 1e-4


def test_place_rules_pick_one_horse_per_race() -> None:
    pred = pd.DataFrame({
        "race_id": [1, 1, 1, 2, 2, 2], "num_pmu": [1, 2, 3, 1, 2, 3],
        "race_date": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 3).date,
        "m": [0.8, 0.5, 0.3, 0.6, 0.55, 0.2],
        "harville_market": [0.85, 0.3, 0.3, 0.7, 0.3, 0.2]})
    payouts = pd.DataFrame({"race_id": [1, 1, 2, 2, 2], "pool": "INTERNET",
                            "bet_type": ["SIMPLE_PLACE"] * 4 + ["SIMPLE_GAGNANT"],
                            "combination": ["1", "2", "2", "3", "2"],
                            "payout_per_eur": [1.1, 2.4, 1.9, 3.0, 4.0]})
    pp = place_payouts(payouts)
    safe = rule_bets(pred, pp, PlaceRule("m", "safe", 0.5))
    assert safe["num_pmu"].tolist() == [1, 1]
    assert safe["profit"].round(2).tolist() == [0.1, -1.0]
    value = rule_bets(pred, pp, PlaceRule("m", "value", 0.2, 1.2))
    # plus fort rapport modèle / marché : n° 2 dans les deux courses
    assert value["num_pmu"].tolist() == [2, 2]
    assert value["profit"].round(2).tolist() == [1.4, 0.9]
    assert all(r.kind == "safe" for r in rule_grid("harville_market"))


def test_place_market_model_round_trip(repo: Repository) -> None:
    from horse_intelligence.intelligence.feature_engineering import (
        CATEGORICAL,
        build_dataset,
        load_raw,
    )
    from horse_intelligence.models.place_model import PlaceMarketModel
    from horse_intelligence.services.training import LGBM_FEATURES, _with_market

    populate(repo, days=24)
    f = build_dataset(load_raw(repo.engine)).frame
    f = _with_market(f[f["won"].notna()])
    dates = sorted(f["race_date"].unique())
    base, cal, test = (f[f["race_date"] < dates[14]],
                       f[f["race_date"].between(dates[14], dates[18])],
                       f[f["race_date"] > dates[18]])
    m = PlaceMarketModel(LGBM_FEATURES, CATEGORICAL).fit(base, cal)
    p = m.predict(test)
    sums = pd.Series(p, index=test.index).groupby(test["race_id"]).sum()
    assert np.allclose(sums, 3.0, atol=1e-6)  # 8 partants → 3 places
    again = PlaceMarketModel.from_dict(m.to_dict()).predict(test)
    np.testing.assert_allclose(again, p)


def test_place_experiment_runs_end_to_end(repo: Repository) -> None:
    from horse_intelligence.services.experiment import PLACE_MODELS, run_place_experiment

    populate(repo, days=36)
    rep = run_place_experiment(repo.engine)
    assert rep["n_folds"] >= 1 and rep["n_races"] > 0
    assert set(rep["metrics"]) == set(PLACE_MODELS)
    assert isinstance(rep["success"], bool)
    assert not rep["errors"]
    for m in rep["metrics"].values():
        assert 0 < m["log_loss"] < 10
