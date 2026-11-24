"""Modèle spécialisé petites courses : définition, combinaison spécifique, walk-forward."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import small_meetings as sm


def test_small_race_definition_and_value_rule() -> None:
    f = pd.DataFrame({"audience": ["NATIONAL", "REGIONAL", "LOCAL", "NATIONAL", None],
                      "prize_eur": [20000, 30000, 5000, 9000, np.nan]})
    assert sm.is_small(f).tolist() == [False, True, True, True, False]
    p = np.array([0.10, 0.10, 0.05, 0.60])
    odds = np.array([15.0, 25.0, 15.0, 1.8])
    # 0,10 × 15 × 0,9 − 1 = +35 % ; cote 25 hors tranche ; 0,05 × 15 × 0,9 − 1 < 0 ;
    # cote 1,8 hors tranche
    assert sm.value_mask(p, odds).tolist() == [True, False, False, False]


def test_context_calibration_adds_small_race_terms() -> None:
    m = sm.ContextLgbm(["x"], [], with_market=True)
    m.raw_score = lambda frame: frame["x"].to_numpy(float)  # type: ignore[method-assign]
    frame = pd.DataFrame({"x": [1.0, 2.0], "mkt_log_prob": [-1.0, -2.0], "small": [1, 0]})
    z = m._calib_inputs(frame)
    assert list(z.columns) == ["lgbm_score", "mkt_log_prob", "small_score", "small_mkt"]
    assert z["small_score"].tolist() == [1.0, 0.0] and z["small_mkt"].tolist() == [-1.0, 0.0]


def test_small_meetings_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import update
    from tests.synthetic import populate

    from horse_intelligence.data.storage import database as db
    from horse_intelligence.services.experiment import run_small_meetings

    populate(repo, days=30)
    with repo.engine.begin() as conn:
        conn.execute(update(db.races).where(db.races.c.course_num <= 2)
                     .values(audience="REGIONAL"))
    rep = run_small_meetings(repo.engine, min_train_days=12, test_days=4, calib_days=6)
    assert rep["n_small"] > 0 and isinstance(rep["success"], bool)
    assert not rep["errors"], rep["errors"]
    assert set(rep["betting"]) == {"lgbm_market", *sm.VARIANTS}
    assert set(rep["variants_vs_prod"]) == set(sm.VARIANTS)
