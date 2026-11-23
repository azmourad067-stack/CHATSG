"""Déferré des 4 (1re fois) × corde : place au départ, filtres, protocole."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import d4_draw


def test_position_uses_published_draw_then_number() -> None:
    f = pd.DataFrame({"draw": [3, np.nan, 12], "num_pmu": [7, 5, 2]})
    assert d4_draw.position(f).tolist() == [3, 5, 12]


def test_protocol_on_synthetic_trot() -> None:
    from tests.test_grand_search import _data

    from horse_intelligence.evaluation.profile_search import prepare

    pred, pays = _data(120)
    f = prepare(pred, pays, "model")
    f = f[f["profit_gagnant"].notna()].copy()
    rng = np.random.default_rng(0)
    f["shoes_d4_first"] = (rng.random(len(f)) < 0.3).astype(float)
    f["draw"] = np.nan
    f["autostart"] = (f["race_id"] % 2).astype(float)
    rep = d4_draw.run(f)
    assert rep["n_strategies"] == 5 * 3 * 4
    assert rep["n_d4"] == int(((f["discipline"] == "trot_attele")
                               & (f["shoes_d4_first"] == 1)).sum())
    assert {"d4", "autres"} == {r["group"] for r in rep["table"]}
    assert isinstance(rep["success"], bool)


def test_d4_draw_experiment_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_d4_draw
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_d4_draw(repo.engine)
    assert isinstance(rep["success"], bool)
    assert "draw" not in rep["missing_columns"] and "autostart" not in rep["missing_columns"]
