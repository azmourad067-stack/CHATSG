from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from tests.conftest import RACE_DAY, START, participants_payload, programme_payload

from horse_intelligence.data.processors.cleaning import canonical_name
from horse_intelligence.data.processors.normalization import (
    entry_from_pmu,
    odds_from_pmu,
    race_from_pmu,
    result_from_pmu,
)
from horse_intelligence.data.processors.validation import DataStatus, validate_snapshot
from horse_intelligence.data.schemas.domain import RaceSnapshot
from horse_intelligence.data.schemas.pmu import PmuParticipants, PmuProgramme


def _snapshot(after_start: bool = True, with_results: bool = True) -> RaceSnapshot:
    prog = PmuProgramme.model_validate(programme_payload())
    reunion = prog.programme.reunions[0]
    race = race_from_pmu(RACE_DAY, reunion, reunion.courses[0])
    parts = PmuParticipants.model_validate(participants_payload(with_results)).participants
    observed = START + (timedelta(hours=1) if after_start else -timedelta(minutes=30))
    snap = RaceSnapshot(race=race, observed_at=observed)
    snap.entries = [entry_from_pmu(p, RACE_DAY) for p in parts]
    snap.odds = [o for p in parts for o in odds_from_pmu(p, snap.observed_after_start)]
    if after_start:
        snap.results = [r for p in parts if (r := result_from_pmu(p))]
    return snap


def test_canonical_name() -> None:
    assert canonical_name("  Éclair  de-Test ") == "ECLAIR DE TEST"
    assert canonical_name("J.-M. Bazire") == "J M BAZIRE"
    assert canonical_name("...") is None


def test_race_normalization() -> None:
    race = _snapshot().race
    assert race.discipline == "trot_attele"
    assert race.scheduled_start_utc == START
    assert race.result_final and race.distance_m == 2700
    assert race.key == "2026-10-29-R1C1"


def test_entry_normalization() -> None:
    e = _snapshot().entries[0]
    assert e.horse_canonical == "ECLAIR DE TEST"
    assert e.birth_year == 2021 and e.rider_name == "D. Pilote1"
    assert e.weight_kg is None  # absent en trot : jamais inventé
    assert e.career_earnings == 85000.0  # centimes → euros
    assert not _snapshot().entries[3].is_runner


def test_snapshot_with_non_runner_is_partial() -> None:
    report = validate_snapshot(_snapshot())
    codes = {c.code for c in report.checks}
    assert "non_runners" in codes
    assert report.status is DataStatus.PARTIAL


def test_missing_start_time_blocks() -> None:
    snap = _snapshot()
    snap.race = replace(snap.race, scheduled_start_utc=None)
    assert validate_snapshot(snap).status is DataStatus.INSUFFICIENT


def test_duplicate_numbers_block() -> None:
    snap = _snapshot()
    snap.entries.append(snap.entries[0])
    assert "duplicate_numbers" in {c.code for c in validate_snapshot(snap).checks}
    assert validate_snapshot(snap).status is DataStatus.INSUFFICIENT


def test_no_entries_blocks() -> None:
    snap = _snapshot()
    snap.entries = []
    assert validate_snapshot(snap).status is DataStatus.INSUFFICIENT


def test_implausible_weight_flags_unit_assumption() -> None:
    snap = _snapshot()
    snap.entries[0] = replace(snap.entries[0], weight_kg=560.0)
    assert "implausible_weight" in {c.code for c in validate_snapshot(snap).checks}


def test_invalid_odds_block() -> None:
    snap = _snapshot()
    snap.odds[0] = replace(snap.odds[0], odds=0.5)
    assert validate_snapshot(snap).status is DataStatus.INSUFFICIENT


def test_result_before_start_is_rejected() -> None:
    snap = _snapshot(after_start=False)
    snap.results = [r for p in PmuParticipants.model_validate(participants_payload()).participants
                    if (r := result_from_pmu(p))]
    assert "result_before_start" in {c.code for c in validate_snapshot(snap).checks}


def test_dead_heat_is_warning_only() -> None:
    snap = _snapshot()
    snap.results[1] = replace(snap.results[1], finish_position=snap.results[0].finish_position)
    report = validate_snapshot(snap)
    assert "dead_heat" in {c.code for c in report.checks}
    assert report.status is DataStatus.PARTIAL


def test_results_units_and_non_runners() -> None:
    snap = _snapshot()
    assert {r.num_pmu for r in snap.results} == {1, 2, 3}  # non-partant exclu
    winner = next(r for r in snap.results if r.finish_position == 1)
    assert winner.time_s == 207.57 and winner.reduction_km_s == 72.6


def test_weight_units() -> None:
    from horse_intelligence.data.processors.normalization import entry_from_pmu
    from horse_intelligence.data.schemas.pmu import PmuParticipant

    p = PmuParticipant.model_validate({"numPmu": 1, "nom": "X", "handicapPoids": 720,
                                       "poidsConditionMonte": 680})
    e = entry_from_pmu(p, RACE_DAY)
    assert (e.weight_kg, e.weight_carried_kg) == (72.0, 68.0)
    p0 = PmuParticipant.model_validate({"numPmu": 1, "nom": "X", "handicapPoids": 580,
                                        "poidsConditionMonte": 0})
    assert entry_from_pmu(p0, RACE_DAY).weight_carried_kg == 58.0


def test_bet_type_and_foreign_earnings() -> None:
    from horse_intelligence.data.processors.normalization import normalize_bet_type
    from horse_intelligence.data.schemas.pmu import PmuParticipant

    assert normalize_bet_type("E_SIMPLE_GAGNANT") == "SIMPLE_GAGNANT"
    assert normalize_bet_type("COUPLE_PLACE") == "COUPLE_PLACE"
    p = PmuParticipant.model_validate({"numPmu": 1, "nom": "X",
                                       "gainsParticipant": {"gainsCarriere": 9460350000}})
    assert entry_from_pmu(p, RACE_DAY, earnings_in_euros=False).career_earnings is None
