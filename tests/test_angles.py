"""Angles de turfiste : définitions, règlement, protocole A / B / C."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import angles


def test_angle_definitions() -> None:
    f = pd.DataFrame({"race_id": [1, 1, 1, 2], "num_pmu": [1, 2, 3, 1],
                      "shoes_d4_first": [1, 0, np.nan, 1],
                      "apt_course_runs": [5, 5, 1, 4], "apt_course_pct": [0.1, 0.6, 0.0, 0.2],
                      "speed_red_mean3": [74.0, 73.2, np.nan, 75.0],
                      "mkt_rank": [1, 2, 3, 1], "mkt_prob": [0.4, 0.3, 0.3, 0.9],
                      "_p_model": [0.3, 0.45, 0.25, 0.9]})
    m = angles.masks(f)
    assert m["déferré des 4 pour la 1re fois"].tolist() == [True, False, False, True]
    # percentile bas = bonnes arrivées ; au moins 3 courses sur l'hippodrome
    assert m["réussit sur cet hippodrome"].tolist() == [True, False, False, True]
    assert m["meilleure vitesse récente du lot"].tolist() == [False, True, False, True]
    assert m["modèle ≥ 1,3 × la cote"].tolist() == [False, True, False, False]
    assert m["favori des cotes"].tolist() == [True, False, False, True]
    assert not m["œillères pour la 1re fois"].any()  # variable absente : aucun cheval


def test_protocol_runs_and_is_fair_on_a_fair_market() -> None:
    from tests.test_grand_search import _data

    from horse_intelligence.evaluation.profile_search import prepare

    pred, pays = _data(120)
    f = prepare(pred, pays, "model")
    f = f[f["profit_gagnant"].notna()].assign(_p_model=f["model"], mkt_rank=f["rank"])
    rep = angles.run(f)
    assert rep["n_strategies"] == len(angles.ANGLES) * 5 * 3
    assert {a["angle"] for a in rep["angles"]} <= set(angles.ANGLES)
    assert all(0 <= a["q"] <= 1 for a in rep["angles"])
    assert isinstance(rep["success"], bool)


def test_angles_experiment_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_angles
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_angles(repo.engine)
    assert isinstance(rep["success"], bool)
    if rep["n_races"]:
        assert rep["missing_columns"] == []  # toutes les variables des angles existent
