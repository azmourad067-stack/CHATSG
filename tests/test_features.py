from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import update
from tests.synthetic import START_DAY, populate

from horse_intelligence.data.storage import database as db
from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.intelligence.feature_engineering import (
    DECISION_OFFSET,
    build_dataset,
    load_raw,
)
from horse_intelligence.intelligence.leakage_audit import audit_source_aggregates
from horse_intelligence.intelligence.musique import musique_features, parse_musique

HISTORY_COLS = ["jockey_win_rate", "jockey_db_runs", "trainer_win_rate", "trainer_db_runs",
                "horse_db_runs", "horse_days_since_db", "horse_db_mean_pct", "pair_db_runs"]


def test_parse_musique() -> None:
    runs = parse_musique("2a0aDa(25)1m7p")
    assert [r.position for r in runs] == [2.0, 10.0, None, 1.0, 7.0]
    assert [r.discipline for r in runs] == ["a", "a", "a", "m", "p"]
    f = musique_features("1a2a3a9a9a9a", "trot_attele")
    assert f["mus_mean_pos_3"] == 2.0 and f["mus_trend"] == 7.0 and f["mus_wins_6"] == 1
    assert f["mus_same_disc_share"] == 1.0
    assert np.isnan(musique_features(None, "plat")["mus_last_pos"])
    assert musique_features(float("nan"), "plat")["mus_n"] == 0  # NaN pandas = aucune course


@pytest.fixture
def synth(repo: Repository) -> Repository:
    populate(repo, days=12)
    return repo


def test_dataset_shape_and_market(synth: Repository) -> None:
    f = build_dataset(load_raw(synth.engine)).frame
    assert f["race_id"].nunique() == 12 * 6
    assert np.allclose(f.groupby("race_id")["mkt_prob"].sum(), 1.0)
    assert (f["mkt_pool"] == "INTERNET").all()
    assert f["won"].sum() == 12 * 6


def test_history_features_ignore_same_day_and_future(synth: Repository) -> None:
    """Permutation du futur : modifier les résultats d'un jour ne change aucune variable
    des courses de ce jour ni des jours antérieurs."""
    raw = load_raw(synth.engine)
    before = build_dataset(raw).frame
    cut = START_DAY + timedelta(days=6)
    altered = {k: v.copy() for k, v in raw.items()}
    res = altered["results"].merge(altered["races"][["race_id", "race_date"]], on="race_id")
    mask = (res["race_date"] >= cut).to_numpy()
    rng = np.random.default_rng(0)
    altered["results"].loc[mask, "finish_position"] = rng.permutation(
        altered["results"].loc[mask, "finish_position"].to_numpy())
    after = build_dataset(altered).frame
    key = ["race_id", "num_pmu"]
    b = before[before["race_date"] <= cut].set_index(key)[HISTORY_COLS].sort_index()
    a = after[after["race_date"] <= cut].set_index(key)[HISTORY_COLS].sort_index()
    pd.testing.assert_frame_equal(a, b)
    # …alors que les jours suivants changent bien (le test a de la puissance)
    b2 = before[before["race_date"] > cut + timedelta(days=1)].set_index(key)[HISTORY_COLS]
    a2 = after[after["race_date"] > cut + timedelta(days=1)].set_index(key)[HISTORY_COLS]
    assert not a2.sort_index().equals(b2.sort_index())


def test_future_races_without_result_do_not_alter_past(synth: Repository) -> None:
    """Une course future sans arrivée ou avec ex aequo ne doit modifier aucune variable
    passée : aucune statistique globale calculée sur tout l'historique."""
    raw = load_raw(synth.engine)
    before = build_dataset(raw).frame
    cut = START_DAY + timedelta(days=6)
    altered = {k: v.copy() for k, v in raw.items()}
    res = altered["results"].merge(altered["races"][["race_id", "race_date"]], on="race_id")
    future = res["race_id"][res["race_date"] > cut].unique()[:10]
    results = altered["results"]
    results.loc[results["race_id"].isin(future), "finish_position"] = np.nan
    dead_heat = res["race_id"][res["race_date"] > cut].unique()[-1]  # deux premiers ex aequo
    results.loc[(results["race_id"] == dead_heat) & (results["finish_position"] == 2),
                "finish_position"] = 1
    after = build_dataset(altered).frame
    key = ["race_id", "num_pmu"]
    b = before[before["race_date"] <= cut].set_index(key)[HISTORY_COLS].sort_index()
    a = after[after["race_date"] <= cut].set_index(key)[HISTORY_COLS].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_first_day_has_no_history(synth: Repository) -> None:
    f = build_dataset(load_raw(synth.engine)).frame
    first = f[f["race_date"] == START_DAY]
    assert (first["jockey_db_runs"] == 0).all() and (first["horse_db_runs"] == 0).all()
    assert first["horse_days_since_db"].isna().all()


def test_odds_published_after_decision_are_ignored(synth: Repository) -> None:
    with synth.engine.begin() as conn:
        rid = conn.execute(db.races.select().limit(1)).first().race_id
        start = conn.execute(db.races.select().where(db.races.c.race_id == rid)).first()
        late = start.scheduled_start_utc - DECISION_OFFSET + timedelta(minutes=1)
        conn.execute(update(db.odds_snapshots).where(db.odds_snapshots.c.race_id == rid)
                     .values(published_at=late))
    f = build_dataset(load_raw(synth.engine)).frame
    assert f.loc[f["race_id"] == rid, "mkt_prob"].isna().all()


def test_non_runners_excluded(synth: Repository) -> None:
    with synth.engine.begin() as conn:
        conn.execute(update(db.race_entries).where(db.race_entries.c.num_pmu == 8)
                     .values(status="NON_PARTANT"))
    f = build_dataset(load_raw(synth.engine)).frame
    assert (f["num_pmu"] != 8).all() and (f["n_runners"] == 7).all()


def test_leakage_audit_detects_post_race_musique() -> None:
    n = 400
    df = pd.DataFrame({"finish_position": np.tile(np.arange(1, 9), n // 8)})
    # honnête : certains gagnants n'avaient encore jamais gagné avant la course
    honest = df.assign(form_string=["2a3a"] * n, career_wins=np.arange(n) % 3)
    assert audit_source_aggregates(honest).passed
    df["career_wins"] = 2
    leaky = df.assign(form_string=np.where(df["finish_position"] == 1, "1a3a", "5a3a"),
                      career_wins=np.where(df["finish_position"] == 1, 1, 0))
    res = audit_source_aggregates(leaky)
    assert not res.passed and res.winner_leading_one == 1.0
