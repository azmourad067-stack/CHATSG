"""Orchestration d'une collecte : programme → partants → rapports → contrôles qualité."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from horse_intelligence.data.collectors.base import CollectorError, NotFoundError
from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.data.processors.normalization import (
    entry_from_pmu,
    odds_from_pmu,
    payouts_from_pmu,
    races_from_programme,
    result_from_pmu,
)
from horse_intelligence.data.processors.validation import DataStatus, validate_snapshot
from horse_intelligence.data.schemas.domain import RaceSnapshot
from horse_intelligence.data.storage.repositories import Repository

logger = logging.getLogger(__name__)


@dataclass
class DayReport:
    day: date
    status: str = "success"
    races_seen: int = 0
    races_ok: int = 0
    statuses: dict[str, str] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)


def ingest_day(day: date, pmu: PmuCollector, repo: Repository,
               fetch_payouts: bool = True) -> DayReport:
    """Collecte une journée. N'interrompt pas la journée sur l'échec d'une course isolée."""
    report = DayReport(day)
    run_id = repo.start_run(day)
    try:
        raw_prog = pmu.fetch_programme_raw(day)
        prog_payload = repo.store_raw(raw_prog)
        programme = pmu.parse_programme(raw_prog)
    except NotFoundError as exc:
        report.status = "failed"
        report.errors.append({"scope": "programme", "message": str(exc)})
        repo.finish_run(run_id, report.status, 0, 0, report.errors)
        return report
    except CollectorError as exc:
        report.status = "failed"
        report.errors.append({"scope": "programme", "message": str(exc)})
        repo.finish_run(run_id, report.status, 0, 0, report.errors)
        raise

    for _reunion, race in races_from_programme(day, programme):
        report.races_seen += 1
        race_id = repo.upsert_race(race, raw_prog.fetched_at, pmu.key, prog_payload)
        # Partants : une réponse par masse de paris. La première réponse obtenue fournit
        # les partants et les résultats ; chaque masse fournit ses propres cotes.
        snap: RaceSnapshot | None = None
        main_payload: int | None = None
        failures: list[str] = []
        for pool in pmu.pools:
            try:
                raw_part = pmu.fetch_participants_raw(day, race.reunion_num, race.course_num,
                                                      pool)
                payload = repo.store_raw(raw_part)
                participants = pmu.parse_participants(raw_part).participants
            except NotFoundError:
                continue
            except CollectorError as exc:
                failures.append(f"{pool}: {exc}")
                continue
            if snap is None:
                snap = RaceSnapshot(race=race, observed_at=raw_part.fetched_at)
                main_payload = payload
                snap.entries = [entry_from_pmu(p, day) for p in participants]
                if snap.observed_after_start:
                    snap.results = [r for p in participants
                                    if (r := result_from_pmu(p)) is not None]
            after = race.scheduled_start_utc is not None \
                and raw_part.fetched_at >= race.scheduled_start_utc
            snap.odds += [o for p in participants
                          for o in odds_from_pmu(p, after, pool, payload)]
        report.errors += [{"scope": race.key, "message": f} for f in failures]
        if snap is None:
            if not failures:
                report.errors.append({"scope": race.key, "message": "partants absents"})
            report.statuses[race.key] = DataStatus.UNAVAILABLE.value
            continue

        if fetch_payouts and race.result_final and snap.observed_after_start:
            for pool in pmu.pools:
                try:
                    raw_rap = pmu.fetch_rapports_raw(day, race.reunion_num, race.course_num,
                                                     pool)
                    rap_payload = repo.store_raw(raw_rap)
                    snap.payouts += payouts_from_pmu(pmu.parse_rapports(raw_rap), pool,
                                                     rap_payload)
                except NotFoundError:
                    continue  # masse sans rapports pour cette course (fréquent à l'étranger)
                except CollectorError as exc:
                    report.errors.append({"scope": f"{race.key}/rapports/{pool}",
                                          "message": str(exc)})

        quality = validate_snapshot(snap)
        repo.save_snapshot(race_id, snap, main_payload)
        repo.save_quality_report(quality)
        report.statuses[race.key] = quality.status.value
        if quality.status is not DataStatus.INSUFFICIENT:
            report.races_ok += 1

    if report.errors:
        report.status = "partial"
    repo.finish_run(run_id, report.status, report.races_seen, report.races_ok, report.errors)
    logger.info("Collecte %s : %d courses, %d exploitables, %d erreur(s)", day,
                report.races_seen, report.races_ok, len(report.errors))
    return report
