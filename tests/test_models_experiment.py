"""Expérience « models » : candidats à LightGBM, de bout en bout sur données synthétiques."""

from __future__ import annotations

import numpy as np
import pytest
from tests.synthetic import populate

from horse_intelligence.data.storage.repositories import Repository

pytest.importorskip("catboost")


def test_rank_score_orders_runners_within_races(repo: Repository) -> None:
    from horse_intelligence.intelligence.feature_engineering import build_dataset, load_raw
    from horse_intelligence.models.challengers import CalibratedScores, LgbmRankScore
    from horse_intelligence.services.training import CATEGORICAL, LGBM_FEATURES

    populate(repo, days=30)
    f = build_dataset(load_raw(repo.engine)).frame
    f = f[f["won"].notna()]
    dates = sorted(f["race_date"].unique())
    base, calib = f[f["race_date"] < dates[20]], f[f["race_date"] >= dates[20]]
    rank = LgbmRankScore(LGBM_FEATURES, CATEGORICAL).fit(base)
    m = CalibratedScores({"rank": rank}, with_market=True).fit(calib)
    p = m.predict(calib)
    sums = calib.assign(p=p).groupby("race_id")["p"].sum()
    assert np.allclose(sums, 1.0)
    # le score porte de l'information : le vainqueur est mieux classé que la moyenne
    assert p[calib["won"].to_numpy() == 1].mean() > p.mean()


def test_models_experiment_runs_end_to_end(repo: Repository) -> None:
    from horse_intelligence.services.experiment import MODEL_CHALLENGERS, run_models_experiment

    populate(repo, days=36)
    rep = run_models_experiment(repo.engine)
    assert rep["n_folds"] >= 1 and rep["n_races"] > 0
    assert set(rep["vs_incumbent"]) == set(MODEL_CHALLENGERS)
    assert isinstance(rep["success"], bool)
    assert rep["catboost_refits"] >= 1
    assert not rep["errors"]
