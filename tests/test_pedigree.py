"""Origines (père, mère, père de la mère) et valeur handicap : collecte et rattrapage."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select, update
from tests.conftest import RACE_DAY, START, make_raw, participants_payload
from tests.test_no_data_leakage import FakePmu

from horse_intelligence.data.storage import database as db
from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.services.ingestion import backfill_pedigree, ingest_day

PEDIGREE_KEYS = ("nomPere", "nomMere", "nomPereMere", "handicapValeur")


class OldPmu(FakePmu):
    """Réponses telles que collectées avant le 15/11/2026 : champs non lus."""

    def fetch_participants_raw(self, day, reunion, course, pool="OFFLINE"):  # type: ignore[no-untyped-def]
        raw = super().fetch_participants_raw(day, reunion, course, pool)
        payload = participants_payload(self.with_results)
        for p in payload["participants"]:
            for k in PEDIGREE_KEYS:
                p.pop(k, None)
        return make_raw("participants", payload, self.observed_at, raw.url)


def _pedigree(repo: Repository) -> dict[str, tuple[str | None, str | None, str | None]]:
    h = db.horses
    with repo.engine.connect() as conn:
        return {r.canonical_name: (r.sire, r.dam, r.dam_sire)
                for r in conn.execute(select(h.c.canonical_name, h.c.sire, h.c.dam,
                                             h.c.dam_sire))}


def _handicaps(repo: Repository) -> list[float | None]:
    e = db.race_entries
    with repo.engine.connect() as conn:
        return sorted(conn.execute(select(e.c.handicap_value)).scalars(),
                      key=lambda v: (v is None, v))


def test_collection_stores_pedigree_and_handicap(repo: Repository) -> None:
    ingest_day(RACE_DAY, FakePmu(START + timedelta(days=1), True), repo)
    ped = _pedigree(repo)
    assert ped["ECLAIR DE TEST"] == ("PERE 1", "MERE 1", "GRAND PERE")
    assert 51.0 in _handicaps(repo) and None not in _handicaps(repo)
    assert repo.races_missing_pedigree(RACE_DAY, RACE_DAY).empty


def test_backfill_fills_old_rows_without_overwriting(repo: Repository) -> None:
    ingest_day(RACE_DAY, OldPmu(START + timedelta(days=1), True), repo)
    assert all(v == (None, None, None) for v in _pedigree(repo).values())
    missing = repo.races_missing_pedigree(RACE_DAY, RACE_DAY)
    assert len(missing) == 1
    # une valeur déjà présente n'est jamais écrasée par le rattrapage
    with repo.engine.begin() as conn:
        conn.execute(update(db.race_entries).where(db.race_entries.c.num_pmu == 2)
                     .values(handicap_value=99.0))
    done, absent, errors = backfill_pedigree(FakePmu(START + timedelta(days=2), True), repo,
                                             RACE_DAY, RACE_DAY, pause_s=0)
    assert (done, absent, errors) == (1, 0, [])
    assert _pedigree(repo)["BOLIDE FICTIF"] == ("PERE 2", "MERE 2", "GRAND PERE")
    assert 99.0 in _handicaps(repo) and 52.0 not in _handicaps(repo)
    assert repo.races_missing_pedigree(RACE_DAY, RACE_DAY).empty
    # reprise : plus rien à faire
    assert backfill_pedigree(FakePmu(START, True), repo, RACE_DAY, RACE_DAY,
                             pause_s=0) == (0, 0, [])
