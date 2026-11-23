"""Signal « déferré des 4 pour la 1re fois » : même définition que la variable testée."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from sqlalchemy import select, update

from horse_intelligence.data.storage import database as db
from horse_intelligence.prediction.signals import D4, first_time_d4


def test_first_time_d4_matches_feature(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import START_DAY, populate

    from horse_intelligence.intelligence.feature_engineering import build_dataset, load_raw

    populate(repo, days=6)
    e = db.race_entries
    with repo.engine.begin() as conn:
        ids = conn.execute(select(e.c.entry_id).order_by(e.c.entry_id)).scalars().all()
        # un tiers des engagements déferrés des 4, réparti sur les jours
        conn.execute(update(e).where(e.c.entry_id.in_(ids[::3])).values(shoes=D4))
    day = START_DAY + timedelta(days=5)
    race_ids = [int(x) for x in repo.races_on(day)["race_id"]]
    got = first_time_d4(repo.engine, race_ids, day)
    frame = build_dataset(load_raw(repo.engine)).frame
    want = frame[frame["race_id"].isin(race_ids) & (frame["shoes_d4_first"] == 1)]
    expected = set(zip(want["race_id"].astype(int), want["num_pmu"].astype(int),
                       strict=True))
    assert got == expected and got  # le test a de la puissance
    # un cheval déjà déferré des 4 à sa course précédente n'est jamais retenu
    hist = frame[frame["race_date"] < day]
    last = hist.sort_values(["race_date", "race_id"]).groupby("horse_id").tail(1)
    already = set(last.loc[last["shoes"] == D4, "horse_id"].astype(int))
    today = frame[frame["race_id"].isin(race_ids)]
    for rid, num in got:
        h = int(today[(today["race_id"] == rid) & (today["num_pmu"] == num)]["horse_id"]
                .iloc[0])
        assert h not in already


def test_no_race_or_no_d4() -> None:
    assert first_time_d4(None, [], pd.Timestamp("2026-01-01").date()) == set()  # type: ignore[arg-type]
