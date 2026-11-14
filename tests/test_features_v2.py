"""Variables v2 (vitesse, ferrure, aptitudes, forme récente) : valeurs et absence de fuite."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from tests.synthetic import START_DAY, populate

from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.intelligence.feature_engineering import (
    D4,
    RECENT_DAYS,
    V2_FEATURES,
    build_dataset,
    load_raw,
)

KEY = ["race_id", "num_pmu"]


@pytest.fixture
def raw(repo: Repository) -> dict[str, pd.DataFrame]:
    populate(repo, days=40)
    r = load_raw(repo.engine)
    # réductions kilométriques synthétiques, liées à l'arrivée (1er = plus rapide)
    r["results"]["reduction_km_s"] = 72.0 + 0.4 * r["results"]["finish_position"]
    return r


def _alter_future(raw: dict[str, pd.DataFrame], cut: object) -> dict[str, pd.DataFrame]:
    altered = {k: v.copy() for k, v in raw.items()}
    res = altered["results"].merge(altered["races"][["race_id", "race_date"]], on="race_id")
    mask = (res["race_date"] >= cut).to_numpy()
    rng = np.random.default_rng(0)
    for c in ("finish_position", "reduction_km_s"):
        altered["results"].loc[mask, c] = rng.permutation(
            altered["results"].loc[mask, c].to_numpy())
    return altered


def test_v2_features_do_not_use_same_day_or_future(raw: dict[str, pd.DataFrame]) -> None:
    cut = START_DAY + timedelta(days=20)
    before = build_dataset(raw).frame
    after = build_dataset(_alter_future(raw, cut)).frame
    b = before[before["race_date"] <= cut].set_index(KEY)[V2_FEATURES].sort_index()
    a = after[after["race_date"] <= cut].set_index(KEY)[V2_FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b)
    # …mais les jours suivants changent bien (le test a de la puissance)
    later = [c for c in V2_FEATURES if not c.startswith(("shoes", "blinkers"))]
    b2 = before[before["race_date"] > cut + timedelta(days=1)].set_index(KEY)[later]
    a2 = after[after["race_date"] > cut + timedelta(days=1)].set_index(KEY)[later]
    assert not a2.sort_index().equals(b2.sort_index())


def test_speed_is_mean_of_last_three_previous_races(raw: dict[str, pd.DataFrame]) -> None:
    f = build_dataset(raw).frame
    horse = f["horse_id"].value_counts().index[0]
    h = f[f["horse_id"] == horse].sort_values("race_date").reset_index(drop=True)
    i = len(h) - 1
    # jours strictement antérieurs ; un même jour, la meilleure réduction compte
    past = h[h["race_date"] < h.loc[i, "race_date"]].assign(
        red=lambda x: 72.0 + 0.4 * x["finish_position"]).groupby("race_date")["red"].min()
    assert h.loc[i, "speed_red_mean3"] == pytest.approx(past.tail(3).mean())
    assert h.loc[i, "speed_runs"] == len(past)
    assert np.isnan(h.loc[0, "speed_red_mean3"]) and h.loc[0, "speed_runs"] == 0
    # comparée au peloton : moyenne nulle dans chaque course
    assert f.groupby("race_id")["speed_red_mean3_rel"].sum().abs().max() < 1e-9


def test_recent_form_counts_previous_thirty_days(raw: dict[str, pd.DataFrame]) -> None:
    f = build_dataset(raw).frame
    row = f[f["race_date"] == f["race_date"].max()].iloc[0]
    d = pd.Timestamp(row["race_date"])
    jockey_dates = pd.to_datetime(f.loc[f["jockey_id"] == row["jockey_id"], "race_date"])
    in_window = ((jockey_dates < d) & (jockey_dates >= d - pd.Timedelta(days=RECENT_DAYS)))
    assert row["jockey_runs_30d"] == in_window.sum()
    assert row["jockey_runs_30d"] < row["jockey_db_runs"]  # la fenêtre exclut le plus ancien


def test_first_time_fully_unshod(raw: dict[str, pd.DataFrame]) -> None:
    f0 = build_dataset(raw).frame
    horse = f0["horse_id"].value_counts().index[0]
    races = f0[f0["horse_id"] == horse].sort_values("race_date")
    r1, r2, r3 = races.iloc[2], races.iloc[3], races.iloc[4]
    e = raw["entries"]

    def set_shoes(r: pd.Series, value: str) -> None:
        e.loc[(e["race_id"] == r["race_id"]) & (e["num_pmu"] == r["num_pmu"]), "shoes"] = value

    set_shoes(r1, "DEFERRE_POSTERIEURS")
    set_shoes(r2, D4)
    set_shoes(r3, D4)
    f = build_dataset(raw).frame.set_index(KEY)
    get = lambda r, c: f.loc[(r["race_id"], r["num_pmu"]), c]  # noqa: E731
    assert get(r2, "shoes_d4") == 1 and get(r2, "shoes_d4_first") == 1
    assert get(r2, "shoes_change") == 1
    assert get(r3, "shoes_d4") == 1 and get(r3, "shoes_d4_first") == 0
    assert get(r3, "shoes_change") == 0


def test_aptitude_counts_previous_runs_on_same_course(raw: dict[str, pd.DataFrame]) -> None:
    f = build_dataset(raw).frame
    horse = f["horse_id"].value_counts().index[0]
    h = f[f["horse_id"] == horse].sort_values("race_date").reset_index(drop=True)
    # toutes les courses synthétiques ont lieu sur le même hippodrome
    expected = [int((h["race_date"] < d).sum()) for d in h["race_date"]]
    assert h["apt_course_runs"].tolist() == expected
    assert h.loc[0, "apt_course_pct"] == pytest.approx(0.5)


def test_features_experiment_runs_end_to_end(repo: Repository) -> None:
    from horse_intelligence.services.experiment import run_features_experiment

    populate(repo, days=36)
    rep = run_features_experiment(repo.engine)
    assert rep["n_folds"] >= 1 and rep["n_races"] > 0
    assert set(rep["v2_vs_v1"]) == {"lgbm_market_v2", "lgbm_form_v2"}
    assert isinstance(rep["success"], bool)
    assert "lgbm_form_v2" in rep["importance"]
    assert not rep["errors"]
