"""Variables v3 (terrain, catégorie, distance, couple, entraîneur × hippodrome)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from tests.synthetic import START_DAY, populate

from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.intelligence.feature_engineering import (
    V3_FEATURES,
    _going_level,
    build_dataset,
    load_raw,
)

KEY = ["race_id", "num_pmu"]
LABELS = ["Bon", "Souple", "Très souple", "Lourd", "PSF STANDARD", "Bon souple"]
PEN = ["3,1", "3,6", "3,9", "4,8", None, "3,3"]


@pytest.fixture
def raw(repo: Repository) -> dict[str, pd.DataFrame]:
    populate(repo, days=40)
    r = load_raw(repo.engine)
    rng = np.random.default_rng(3)
    ids = r["races"]["race_id"].to_numpy()
    pick = rng.integers(0, len(LABELS), len(ids))
    r["going"] = pd.DataFrame({
        "race_id": ids, "going_label": [LABELS[i] for i in pick],
        "penetrometer": [PEN[i] for i in pick],
        "observed_at": datetime(2026, 1, 1, tzinfo=UTC)})
    # dotations variées pour donner du sens au changement de catégorie
    r["races"]["prize_eur"] = rng.choice([8_000.0, 20_000.0, 50_000.0], len(ids))
    return r


def test_going_level_mapping() -> None:
    assert _going_level("Bon", np.nan) == 1
    assert _going_level("Bon souple", np.nan) == 2
    assert _going_level("Très souple", np.nan) == 3
    assert _going_level("Très lourd", np.nan) == 4
    assert _going_level(None, 3.6) == 2
    assert np.isnan(_going_level(None, np.nan))


def test_v3_features_do_not_use_same_day_or_future(raw: dict[str, pd.DataFrame]) -> None:
    cut = START_DAY + timedelta(days=20)
    before = build_dataset(raw).frame
    altered = {k: v.copy() for k, v in raw.items()}
    res = altered["results"].merge(altered["races"][["race_id", "race_date"]], on="race_id")
    mask = (res["race_date"] >= cut).to_numpy()
    altered["results"].loc[mask, "finish_position"] = np.random.default_rng(0).permutation(
        altered["results"].loc[mask, "finish_position"].to_numpy())
    after = build_dataset(altered).frame
    b = before[before["race_date"] <= cut].set_index(KEY)[V3_FEATURES].sort_index()
    a = after[after["race_date"] <= cut].set_index(KEY)[V3_FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b)
    hist = ["apt_going_pct", "apt_pair_pct", "trainer_course_win_rate"]
    b2 = before[before["race_date"] > cut + timedelta(days=1)].set_index(KEY)[hist]
    a2 = after[after["race_date"] > cut + timedelta(days=1)].set_index(KEY)[hist]
    assert not a2.sort_index().equals(b2.sort_index())


def test_going_class_and_distance_change(raw: dict[str, pd.DataFrame]) -> None:
    f = build_dataset(raw).frame
    assert f["going_psf"].isin([0.0, 1.0]).all()
    assert f.loc[f["going_psf"] == 1, "going_level"].isna().all()
    assert set(f["going_level"].dropna().unique()) <= {1.0, 2.0, 3.0, 4.0}
    horse = f["horse_id"].value_counts().index[0]
    h = f[f["horse_id"] == horse].sort_values(["race_date", "race_id"]).reset_index(drop=True)
    i = len(h) - 1
    prev = h[h["race_date"] < h.loc[i, "race_date"]]
    by_day = prev.groupby("race_date")["log_prize"].mean()
    assert h.loc[i, "class_change"] == pytest.approx(h.loc[i, "log_prize"]
                                                     - by_day.tail(3).mean())
    assert h.loc[i, "dist_change"] == 0  # toutes les courses synthétiques font 2 700 m
    assert np.isnan(h.loc[0, "class_change"])


def test_trainer_course_counts_previous_runs(raw: dict[str, pd.DataFrame]) -> None:
    f = build_dataset(raw).frame
    row = f[f["race_date"] == f["race_date"].max()].iloc[0]
    same = f[(f["trainer_id"] == row["trainer_id"]) & (f["race_date"] < row["race_date"])]
    assert row["trainer_course_runs"] == len(same)  # un seul hippodrome synthétique
