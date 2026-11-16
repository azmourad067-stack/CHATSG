"""Variables v4 (origines, valeur handicap) : valeurs et absence de fuite."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from tests.synthetic import START_DAY, populate

from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.intelligence.feature_engineering import (
    PRIOR_WIN_RATE,
    SHRINK_RUNS,
    V4_FEATURES,
    build_dataset,
    load_raw,
)

KEY = ["race_id", "num_pmu"]


@pytest.fixture
def raw(repo: Repository) -> dict[str, pd.DataFrame]:
    populate(repo, days=40)
    r = load_raw(repo.engine)
    rng = np.random.default_rng(4)
    h = r["horses"]
    # 6 pères et 4 pères de mère fictifs, une valeur handicap par engagement
    h["sire"] = [f"PERE {i % 6}" for i in range(len(h))]
    h["dam_sire"] = [f"GP {i % 4}" for i in range(len(h))]
    r["entries"]["handicap_value"] = rng.uniform(40, 70, len(r["entries"])).round(1)
    return r


def test_v4_history_features_do_not_use_same_day_or_future(raw: dict[str, pd.DataFrame]
                                                           ) -> None:
    cut = START_DAY + timedelta(days=20)
    before = build_dataset(raw, with_v4=True).frame
    altered = {k: v.copy() for k, v in raw.items()}
    res = altered["results"].merge(altered["races"][["race_id", "race_date"]], on="race_id")
    mask = (res["race_date"] >= cut).to_numpy()
    altered["results"].loc[mask, "finish_position"] = np.random.default_rng(0).permutation(
        altered["results"].loc[mask, "finish_position"].to_numpy())
    after = build_dataset(altered, with_v4=True).frame
    hist = [c for c in V4_FEATURES if not c.startswith("handicap")]
    b = before[before["race_date"] <= cut].set_index(KEY)[hist].sort_index()
    a = after[after["race_date"] <= cut].set_index(KEY)[hist].sort_index()
    pd.testing.assert_frame_equal(a, b)
    b2 = before[before["race_date"] > cut + timedelta(days=1)].set_index(KEY)[hist]
    a2 = after[after["race_date"] > cut + timedelta(days=1)].set_index(KEY)[hist]
    assert not a2.sort_index().equals(b2.sort_index())  # le test a de la puissance


def test_sire_rate_counts_previous_offspring_runs(raw: dict[str, pd.DataFrame]) -> None:
    f = build_dataset(raw, with_v4=True).frame
    row = f[f["race_date"] == f["race_date"].max()].iloc[0]
    past = f[(f["sire"] == row["sire"]) & (f["race_date"] < row["race_date"])
             & f["has_result"].astype(bool)]
    assert row["sire_runs"] == len(past)
    expected = (past["won"].sum() + SHRINK_RUNS * PRIOR_WIN_RATE) / (len(past) + SHRINK_RUNS)
    assert row["sire_win_rate"] == pytest.approx(expected)
    first = f.sort_values("race_date").groupby("sire").head(1)
    assert (first["sire_runs"] == 0).all()


def test_handicap_value_relative_and_rank(raw: dict[str, pd.DataFrame]) -> None:
    f = build_dataset(raw, with_v4=True).frame
    assert f.groupby("race_id")["handicap_value_rel"].sum().abs().max() < 1e-9
    g = f[f["race_id"] == f["race_id"].iloc[0]].sort_values("handicap_value", ascending=False)
    assert g["handicap_rank_pct"].iloc[0] == 0.0 and g["handicap_rank_pct"].iloc[-1] == 1.0


def test_v4_absent_without_pedigree(repo: Repository) -> None:
    populate(repo, days=20)
    f = build_dataset(load_raw(repo.engine), with_v4=True).frame
    assert (f["sire_runs"] == 0).all() and f["handicap_value"].isna().all()
    # par défaut (application, entraînement) : variables v4 non calculées
    assert "sire_runs" not in build_dataset(load_raw(repo.engine)).frame


def test_features4_experiment_runs_end_to_end(repo: Repository) -> None:
    from horse_intelligence.services.experiment import run_features_experiment

    populate(repo, days=36)
    rep = run_features_experiment(repo.engine, stage="v4")
    assert set(rep["new_vs_old"]) == {"lgbm_market_v4", "lgbm_form_v4"}
    assert isinstance(rep["success"], bool) and not rep["errors"]
