from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from horse_intelligence.config.settings import Settings, load_sources
from horse_intelligence.data.collectors.base import RawResponse
from horse_intelligence.data.storage.database import init_db, make_engine
from horse_intelligence.data.storage.raw_store import RawStore
from horse_intelligence.data.storage.repositories import Repository

RACE_DAY = date(2026, 10, 29)
# Départ de R1C1 dans la fixture : 2026-10-29 13:50 UTC
START_MS = int(datetime(2026, 10, 29, 13, 50, tzinfo=UTC).timestamp() * 1000)
START = datetime(2026, 10, 29, 13, 50, tzinfo=UTC)


def programme_payload() -> dict[str, Any]:
    return {"programme": {"date": START_MS, "reunions": [{
        "numOfficiel": 1,
        "hippodrome": {"code": "TST", "libelleCourt": "TESTVILLE",
                       "libelleLong": "Hippodrome de Testville"},
        "pays": {"code": "FRA"},
        "courses": [
            {"numReunion": 1, "numOrdre": 1, "libelle": "PRIX DE TEST", "heureDepart": START_MS,
             "distance": 2700, "distanceUnit": "METRE", "discipline": "ATTELE",
             "specialite": "TROT_ATTELE", "corde": "CORDE_GAUCHE", "montantPrix": 30000,
             "nombreDeclaresPartants": 4, "statut": "FIN_COURSE", "arriveeDefinitive": True,
             "penetrometre": {"valeurMesure": "3,2", "intitule": "Bon"}},
            {"numReunion": 1, "numOrdre": 2, "libelle": "PRIX SANS PARTANTS",
             "heureDepart": START_MS + 1_800_000, "distance": 2100, "specialite": "TROT_ATTELE",
             "nombreDeclaresPartants": 10, "statut": "PROGRAMMEE"},
        ]}]}}


def participants_payload(with_results: bool = True) -> dict[str, Any]:
    def p(num: int, nom: str, place: int | None, ref: float, direct: float,
          statut: str = "PARTANT") -> dict[str, Any]:
        d: dict[str, Any] = {
            "numPmu": num, "nom": nom, "age": 5, "sexe": "HONGRES", "race": "TROTTEUR FRANCAIS",
            "statut": statut, "driver": f"D. Pilote{num}", "entraineur": f"E. Coach{num}",
            "proprietaire": "Écurie Test", "musique": "1a3aDa", "nombreCourses": 20,
            "nombreVictoires": 3, "nombrePlaces": 8,
            "gainsParticipant": {"gainsCarriere": 8500000}, "handicapDistance": 2700,
            "deferre": "DEFERRE_ANTERIEURS_POSTERIEURS",
            "dernierRapportReference": {"typePari": "SIMPLE_GAGNANT", "rapport": ref,
                                        "typeRapport": "REFERENCE",
                                        "dateRapport": START_MS - 30 * 60_000},
            "dernierRapportDirect": {"typePari": "SIMPLE_GAGNANT", "rapport": direct,
                                     "typeRapport": "DIRECT",
                                     "dateRapport": START_MS + 60_000},
        }
        if with_results and place is not None:
            d["ordreArrivee"] = place
            d["tempsObtenu"] = 206570 + 1000 * place
            d["reductionKilometrique"] = 72500 + 100 * place
        if statut == "NON_PARTANT":
            d["incident"] = "NON_PARTANT"
        return d

    return {"participants": [
        p(1, "ÉCLAIR DE TEST", 2, 3.5, 3.1),
        p(2, "BOLIDE FICTIF", 1, 6.0, 7.2),
        p(3, "CHEVAL IMAGINAIRE", 3, 12.0, 9.8),
        p(4, "ABSENT DU JOUR", None, 25.0, 30.0, statut="NON_PARTANT"),
    ]}


def make_raw(endpoint: str, payload: Any, fetched_at: datetime, url: str = "") -> RawResponse:
    return RawResponse(source="pmu", endpoint=endpoint, url=url or f"https://example.test/{endpoint}",
                       params={}, fetched_at=fetched_at, http_status=200,
                       body=json.dumps(payload).encode())


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, database_url=f"sqlite:///{tmp_path / 'test.db'}",
                    sources=load_sources())


@pytest.fixture
def repo(settings: Settings) -> Repository:
    engine = make_engine(settings.database_url)
    init_db(engine)
    r = Repository(engine, RawStore(settings.raw_dir))
    r.sync_sources(settings)
    return r
