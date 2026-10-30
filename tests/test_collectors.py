from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest
import requests
import responses
from tests.conftest import RACE_DAY, make_raw, participants_payload, programme_payload

from horse_intelligence.config.settings import load_sources
from horse_intelligence.data.collectors.base import (
    ContractError,
    HttpClient,
    NotFoundError,
    NotOperationalError,
    RateLimiter,
    SourceStatus,
    SourceUnavailableError,
)
from horse_intelligence.data.collectors.pmu_collector import PmuCollector, pmu_date
from horse_intelligence.data.collectors.registry import build_collectors
from horse_intelligence.data.collectors.weather_collector import WeatherCollector

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/1"


def _pmu(max_retries: int = 2) -> PmuCollector:
    cfg = replace(load_sources()["pmu"], min_interval_s=0.0, max_retries=max_retries)
    client = HttpClient(cfg, "test-agent", sleep=lambda _s: None)
    return PmuCollector(cfg, client=client)


def test_pmu_date_format() -> None:
    assert pmu_date(date(2026, 3, 7)) == "07032026"


@responses.activate
def test_programme_fetch_and_parse() -> None:
    responses.get(f"{BASE}/programme/29102026", json=programme_payload())
    pmu = _pmu()
    raw = pmu.fetch_programme_raw(RACE_DAY)
    prog = pmu.parse_programme(raw)
    assert raw.http_status == 200 and len(raw.sha256) == 64
    assert prog.programme.reunions[0].hippodrome.code == "TST"
    assert [c.num_ordre for c in prog.programme.reunions[0].courses] == [1, 2]
    assert responses.calls[0].request.headers["User-Agent"] == "test-agent"


@responses.activate
def test_retry_then_success() -> None:
    url = f"{BASE}/programme/29102026"
    responses.get(url, status=503)
    responses.get(url, json=programme_payload())
    assert _pmu().fetch_programme_raw(RACE_DAY).http_status == 200
    assert len(responses.calls) == 2


@responses.activate
def test_retries_are_bounded() -> None:
    responses.get(f"{BASE}/programme/29102026", status=502)
    with pytest.raises(SourceUnavailableError, match="3 tentatives"):
        _pmu(max_retries=2).fetch_programme_raw(RACE_DAY)
    assert len(responses.calls) == 3


@responses.activate
def test_connection_error_is_reported() -> None:
    responses.get(f"{BASE}/programme/29102026", body=requests.ConnectionError("boom"))
    with pytest.raises(SourceUnavailableError):
        _pmu(max_retries=0).fetch_programme_raw(RACE_DAY)


@responses.activate
def test_404_is_not_found_without_retry() -> None:
    responses.get(f"{BASE}/programme/29102026/R1/C9/participants", status=404)
    with pytest.raises(NotFoundError):
        _pmu().fetch_participants_raw(RACE_DAY, 1, 9)
    assert len(responses.calls) == 1


@responses.activate
def test_client_error_not_retried() -> None:
    responses.get(f"{BASE}/programme/29102026", status=403)
    with pytest.raises(SourceUnavailableError, match="403"):
        _pmu().fetch_programme_raw(RACE_DAY)
    assert len(responses.calls) == 1


def test_contract_error_on_schema_drift() -> None:
    raw = make_raw("participants", {"partants": []}, fetched_at=datetime.now(UTC))
    with pytest.raises(ContractError):
        PmuCollector.parse_participants(raw)


def test_contract_error_on_non_json() -> None:
    raw = replace(make_raw("programme", {}, datetime.now(UTC)), body=b"<html>")
    with pytest.raises(ContractError):
        PmuCollector.parse_programme(raw)


def test_participants_parse_fixture() -> None:
    parsed = PmuCollector.parse_participants(make_raw("participants", participants_payload(),
                                                      datetime.now(UTC)))
    assert len(parsed.participants) == 4
    assert parsed.participants[0].dernier_rapport_reference is not None


@responses.activate
def test_health_reports_unavailable() -> None:
    responses.get(f"{BASE}/programme/{pmu_date(date.today())}", status=500)
    assert _pmu(max_retries=0).health().status is SourceStatus.UNAVAILABLE


def test_disabled_sources_are_not_operational(settings) -> None:  # type: ignore[no-untyped-def]
    collectors = build_collectors(settings)
    assert collectors["turf_bzh"].health().status is SourceStatus.NOT_OPERATIONAL
    cfg = replace(load_sources()["pmu"], enabled=False)
    with pytest.raises(NotOperationalError):
        PmuCollector(cfg).fetch_programme_raw(RACE_DAY)


def test_rate_limiter_waits() -> None:
    t = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        t[0] += s

    rl = RateLimiter(1.0, clock=lambda: t[0], sleep=sleep)
    rl.wait()
    t[0] += 0.25
    rl.wait()
    assert slept == [pytest.approx(0.75)]


@responses.activate
def test_weather_parse() -> None:
    cfg = replace(load_sources()["open_meteo"], min_interval_s=0.0)
    responses.get("https://api.open-meteo.com/v1/forecast", json={"hourly": {
        "time": ["2026-10-29T13:00", "2026-10-29T14:00"],
        "temperature_2m": [12.5, 13.0], "precipitation": [0.0, 0.4],
        "wind_speed_10m": [3.1, 4.0]}})
    w = WeatherCollector(cfg, client=HttpClient(cfg, "t", sleep=lambda _s: None))
    points = w.forecast(48.8, 2.4, RACE_DAY)
    assert len(points) == 2 and points[1].precip_mm == 0.4 and points[0].kind == "forecast"
    assert responses.calls[0].request.params["wind_speed_unit"] == "ms"


# --- Tests de contrat contre la vraie API (pytest -m network) ----------------------
@pytest.mark.network
def test_live_pmu_contract() -> None:
    pmu = PmuCollector(load_sources()["pmu"])
    prog = pmu.parse_programme(pmu.fetch_programme_raw(date.today()))
    reunion = prog.programme.reunions[0]
    course = reunion.courses[0]
    parts = pmu.parse_participants(
        pmu.fetch_participants_raw(date.today(), reunion.num_officiel, course.num_ordre))
    assert parts.participants and course.heure_depart is not None
