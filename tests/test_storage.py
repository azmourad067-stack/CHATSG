from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from tests.conftest import RACE_DAY, START, make_raw
from tests.test_no_data_leakage import FakePmu

from horse_intelligence.data.storage import database as db
from horse_intelligence.data.storage.parquet_manager import export_tables
from horse_intelligence.data.storage.raw_store import RawStore
from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.services.ingestion import ingest_day


def _count(repo: Repository, table) -> int:  # type: ignore[no-untyped-def]
    with repo.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def test_ingest_day_end_to_end(repo: Repository) -> None:
    report = ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), repo)
    assert report.races_seen == 2 and report.races_ok == 1
    assert report.status == "partial"  # R1C2 sans partants
    assert report.statuses["2026-10-29-R1C2"] == "unavailable"
    counts = repo.counts()
    assert counts["races"] == 2 and counts["horses"] == 4 and counts["results"] == 3
    assert _count(repo, db.payouts) == 1
    races = repo.races_on(RACE_DAY)
    assert races.set_index("course_num").loc[1, "data_status"] == "partial"


def test_reingest_is_idempotent(repo: Repository) -> None:
    fake = FakePmu(START + timedelta(hours=2), True)
    ingest_day(RACE_DAY, fake, repo)
    n_entries, n_odds = _count(repo, db.race_entries), _count(repo, db.odds_snapshots)
    ingest_day(RACE_DAY, fake, repo)
    assert _count(repo, db.race_entries) == n_entries
    assert _count(repo, db.odds_snapshots) == n_odds
    assert repo.counts()["horses"] == 4


def test_raw_payloads_are_traceable(repo: Repository) -> None:
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), repo)
    with repo.engine.connect() as conn:
        rows = conn.execute(select(db.raw_payloads)).mappings().all()
        entry_payloads = conn.execute(select(db.race_entries.c.payload_id).distinct()).all()
    assert rows and all(len(r["sha256"]) == 64 for r in rows)
    payload_ids = {r["payload_id"] for r in rows}
    assert {p[0] for p in entry_payloads} <= payload_ids
    assert RawStore.read(__import__("pathlib").Path(rows[0]["storage_uri"]))


def test_naive_datetime_rejected(repo: Repository) -> None:
    raw = make_raw("programme", {}, datetime(2026, 1, 1))
    with pytest.raises(Exception, match="naïf"):
        repo.store_raw(raw)


def test_day_completed(repo: Repository) -> None:
    assert not repo.day_completed(RACE_DAY)
    run = repo.start_run(RACE_DAY)
    repo.finish_run(run, "success", 1, 1, [])
    # la collecte a démarré aujourd'hui, donc après la fin de la journée cible
    assert repo.day_completed(RACE_DAY) == (datetime.now(UTC).date() > RACE_DAY)


def test_horse_search_and_history(repo: Repository) -> None:
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), repo)
    found = repo.search_horses("bolide")
    assert list(found["display_name"]) == ["BOLIDE FICTIF"]
    hist = repo.horse_history(int(found["horse_id"].iloc[0]))
    assert hist["finish_position"].tolist() == [1]


def test_parquet_export(repo: Repository, tmp_path) -> None:  # type: ignore[no-untyped-def]
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), repo)
    written = export_tables(repo.engine, tmp_path / "pq", ["races", "race_entries"])
    import pandas as pd
    assert len(pd.read_parquet(written["races"])) == 2
