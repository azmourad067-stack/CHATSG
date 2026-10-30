"""Garde-fous anti-fuite de la couche de données (bloquants en CI)."""

from __future__ import annotations

from datetime import timedelta

from tests.conftest import (
    RACE_DAY,
    START,
    make_raw,
    participants_payload,
    programme_payload,
)

from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.data.processors.normalization import odds_from_pmu
from horse_intelligence.data.schemas.pmu import PmuParticipants
from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.services.ingestion import ingest_day


class FakePmu(PmuCollector):
    """Rejoue des réponses figées à un instant d'observation contrôlé."""

    def __init__(self, observed_at, with_results: bool,  # type: ignore[no-untyped-def]
                 pools: tuple[str, ...] = ("OFFLINE",)) -> None:
        self.observed_at = observed_at
        self.with_results = with_results
        self._pools = pools

    @property
    def pools(self) -> tuple[str, ...]:
        return self._pools

    def fetch_programme_raw(self, day):  # type: ignore[no-untyped-def]
        return make_raw("programme", programme_payload(), self.observed_at, "u/prog")

    def fetch_participants_raw(self, day, reunion, course, pool="OFFLINE"):  # type: ignore[no-untyped-def]
        if course != 1:
            from horse_intelligence.data.collectors.base import NotFoundError
            raise NotFoundError("absent")
        payload = participants_payload(self.with_results)
        if pool == "INTERNET":  # masse Internet : cotes différentes
            for p in payload["participants"]:
                p["dernierRapportReference"]["rapport"] += 1.0
        return make_raw("participants", payload, self.observed_at,
                        f"u/R{reunion}C{course}/{pool}")

    def fetch_rapports_raw(self, day, reunion, course, pool="OFFLINE"):  # type: ignore[no-untyped-def]
        return make_raw("rapports-definitifs", [{"typePari": "SIMPLE_GAGNANT", "rapports": [
            {"combinaison": "2", "dividendePourUnEuro": 710}]}], self.observed_at, "u/rap")


def _race_id(repo: Repository) -> int:
    races = repo.races_on(RACE_DAY)
    return int(races[races["course_num"] == 1]["race_id"].iloc[0])


def test_direct_odds_after_start_are_post_start() -> None:
    p = PmuParticipants.model_validate(participants_payload()).participants[0]
    kinds = {o.odds_kind for o in odds_from_pmu(p, observed_after_start=True)}
    assert kinds == {"reference", "post_start"}
    kinds = {o.odds_kind for o in odds_from_pmu(p, observed_after_start=False)}
    assert kinds == {"reference", "live"}


def test_backfilled_day_never_exposes_post_start_odds(repo: Repository) -> None:
    # Collecte d'une journée passée : tout est observé après le départ.
    ingest_day(RACE_DAY, FakePmu(START + timedelta(days=1), True), repo)
    race_id = _race_id(repo)
    odds = repo.odds_as_of(race_id, START + timedelta(days=2),
                           kinds=("reference", "live", "post_start"))
    assert set(odds["odds_kind"]) == {"reference"}
    # La cote de référence, publiée 30 min avant le départ, est disponible à H-5…
    at_decision = repo.odds_as_of(race_id, START - timedelta(minutes=5))
    assert set(at_decision["odds_kind"]) == {"reference"}
    assert (at_decision["available_at"] < START).all()
    # …mais pas à H-60, avant sa publication.
    assert repo.odds_as_of(race_id, START - timedelta(minutes=60)).empty
    entries = repo.entries_as_of(race_id)
    assert entries["observed_after_start"].all()


def test_as_of_query_ignores_later_observations(repo: Repository) -> None:
    before = START - timedelta(minutes=30)
    after = START + timedelta(hours=2)
    ingest_day(RACE_DAY, FakePmu(before, with_results=False), repo)
    ingest_day(RACE_DAY, FakePmu(after, with_results=True), repo)
    race_id = _race_id(repo)

    at_decision = repo.entries_as_of(race_id, as_of=START - timedelta(minutes=5))
    assert not at_decision.empty
    assert (at_decision["observed_at"] <= START).all()
    assert not at_decision["observed_after_start"].any()

    odds = repo.odds_as_of(race_id, START - timedelta(minutes=5))
    assert set(odds["odds_kind"]) == {"reference", "live"}
    assert (odds["available_at"] <= START).all()


def test_results_only_stored_after_start(repo: Repository) -> None:
    ingest_day(RACE_DAY, FakePmu(START - timedelta(minutes=10), with_results=True), repo)
    assert repo.results_for(_race_id(repo)).empty


def test_future_changes_do_not_alter_past_view(repo: Repository) -> None:
    """Permutation du futur : une observation postérieure ne modifie pas la vue passée."""
    before = START - timedelta(minutes=30)
    ingest_day(RACE_DAY, FakePmu(before, with_results=False), repo)
    race_id = _race_id(repo)
    view_1 = repo.entries_as_of(race_id, as_of=before)
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=3), with_results=True), repo)
    view_2 = repo.entries_as_of(race_id, as_of=before)
    assert view_1.equals(view_2)


def test_pools_are_kept_separate(repo: Repository) -> None:
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True, ("OFFLINE", "INTERNET")),
               repo)
    race_id = _race_id(repo)
    odds = repo.odds_as_of(race_id, START - timedelta(minutes=5))
    by_pool = odds.set_index(["num_pmu", "pool"])["odds"]
    assert by_pool[(1, "OFFLINE")] == 3.5 and by_pool[(1, "INTERNET")] == 4.5
    assert set(repo.odds_as_of(race_id, START, pool="INTERNET")["pool"]) == {"INTERNET"}
    assert len(repo.entries_as_of(race_id)) == 4  # partants non dupliqués
