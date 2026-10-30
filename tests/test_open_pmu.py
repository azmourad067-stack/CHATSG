"""Connecteur Open PMU et recoupement des arrivées (réponses simulées)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
import responses
from tests.conftest import RACE_DAY, START
from tests.test_no_data_leakage import FakePmu

from horse_intelligence.config.settings import load_sources
from horse_intelligence.data.collectors.base import ContractError, HttpClient, NotFoundError
from horse_intelligence.data.collectors.secondary_collectors import OpenPmuCollector
from horse_intelligence.data.schemas.open_pmu import OpenPmuRace
from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.services.crosscheck import compare_arrivals
from horse_intelligence.services.ingestion import ingest_day

URL = "https://open-pmu-api.vercel.app/api/arrivees"


def _collector() -> OpenPmuCollector:
    cfg = replace(load_sources()["open_pmu_api"], min_interval_s=0.0)
    return OpenPmuCollector(cfg, client=HttpClient(cfg, "t", sleep=lambda _s: None))


def _race(rc: str, arrivee: list[int]) -> dict[str, object]:
    return {"type": "Attelé", "distance": 2700, "prix": "PRIX DE TEST", "lieu": "Testville",
            "heure_depart": "13:50:00", "partants": 4, "non_partants": 0, "arrivee": arrivee,
            "r/c": rc, "date": "2026-10-29",
            "arrivee_details": {"2": {"nom_cheval": "BOLIDE FICTIF", "position": 1,
                                      "cotes": [7.0, 6.5, 7.2]}}}


@responses.activate
def test_fetch_arrivees_uses_us_date_format() -> None:
    responses.get(URL, json={"error": False, "total": 1, "params": {},
                             "message": [_race("R1/C1", [2, 1, 3])]})
    races = _collector().fetch_arrivees(RACE_DAY)
    assert responses.calls[0].request.params["date"] == "10/29/2026"
    assert races[0].reunion_course_nums == (1, 1)
    assert races[0].non_partants == []  # entier hérité converti en liste vide


@responses.activate
def test_error_message_is_not_found() -> None:
    responses.get(URL, json={"error": True, "total": 0, "params": {},
                             "message": "Aucune course trouvée."})
    with pytest.raises(NotFoundError):
        _collector().fetch_arrivees(RACE_DAY)


@responses.activate
def test_schema_drift_is_contract_error() -> None:
    responses.get(URL, json={"unexpected": True})
    with pytest.raises(ContractError):
        _collector().fetch_arrivees(RACE_DAY)


def test_reunion_course_parsing() -> None:
    assert OpenPmuRace.model_validate({"r/c": "R4/C12"}).reunion_course_nums == (4, 12)
    assert OpenPmuRace.model_validate({"r/c": "?"}).reunion_course_nums is None


def test_compare_arrivals(repo: Repository) -> None:
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), repo)
    # Fixture PMU : arrivée 2-1-3
    checks = {c.race_key: c for c in compare_arrivals(repo, RACE_DAY, [
        OpenPmuRace.model_validate(_race("R1/C1", [2, 1, 3])),
        OpenPmuRace.model_validate(_race("R1/C2", [5, 4])),
        OpenPmuRace.model_validate(_race("R9/C9", [1])),
    ])}
    assert checks["2026-10-29-R1C1"].status == "agree"
    assert checks["2026-10-29-R1C2"].status == "missing_primary"
    assert checks["2026-10-29-R9C9"].status == "missing_primary"
    bad = compare_arrivals(repo, RACE_DAY, [OpenPmuRace.model_validate(_race("R1/C1", [1, 2, 3]))])
    assert bad[0].status == "disagree"
