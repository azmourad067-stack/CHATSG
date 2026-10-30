"""Accès aux données : écriture traçable et lectures « as-of »."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine, Table, and_, func, insert, select, update
from sqlalchemy.engine import Connection, CursorResult

from horse_intelligence.config.settings import Settings
from horse_intelligence.data.collectors.base import HealthReport, RawResponse
from horse_intelligence.data.processors.cleaning import canonical_name
from horse_intelligence.data.processors.validation import QualityReport
from horse_intelligence.data.schemas.domain import EntryRecord, RaceRecord, RaceSnapshot
from horse_intelligence.data.storage import database as db
from horse_intelligence.data.storage.raw_store import RawStore

_ENTRY_HASH_FIELDS = (
    "horse_canonical", "rider_name", "trainer_name", "owner_name", "draw", "weight_kg",
    "handicap_distance_m", "blinkers", "shoes", "status", "form_string", "career_starts",
    "career_wins", "career_places", "career_earnings",
)


def _entry_hash(e: EntryRecord) -> str:
    payload = {k: getattr(e, k) for k in _ENTRY_HASH_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _pk(result: CursorResult[Any]) -> int:
    key = result.inserted_primary_key
    if key is None:
        raise RuntimeError("clé primaire non renvoyée par l'insertion")
    return int(key[0])


class Repository:
    def __init__(self, engine: Engine, raw_store: RawStore) -> None:
        self.engine = engine
        self.raw_store = raw_store

    # ------------------------------------------------------------------ sources
    def sync_sources(self, settings: Settings) -> None:
        with self.engine.begin() as conn:
            for key, src in settings.sources.items():
                values = {"name": src.name, "base_url": src.base_url, "kind": src.kind,
                          "enabled": src.enabled, "status_audit": src.status_audit}
                exists = conn.execute(select(db.data_sources.c.source_key).where(
                    db.data_sources.c.source_key == key)).first()
                if exists:
                    conn.execute(update(db.data_sources)
                                 .where(db.data_sources.c.source_key == key).values(**values))
                else:
                    conn.execute(insert(db.data_sources).values(source_key=key, **values))

    def record_health(self, report: HealthReport) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(db.data_sources)
                         .where(db.data_sources.c.source_key == report.source)
                         .values(last_status=report.status.value,
                                 last_checked_at=report.checked_at,
                                 last_message=report.message))

    # ---------------------------------------------------------------- brut
    def store_raw(self, raw: RawResponse) -> int:
        path = self.raw_store.write(raw)
        with self.engine.begin() as conn:
            res = conn.execute(insert(db.raw_payloads).values(
                source_key=raw.source, endpoint=raw.endpoint, url=raw.url,
                params_json=raw.params, fetched_at=raw.fetched_at, http_status=raw.http_status,
                sha256=raw.sha256, size_bytes=len(raw.body), storage_uri=str(path),
            ))
            return _pk(res)

    # ---------------------------------------------------------------- runs
    def start_run(self, target_date: date) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(insert(db.collection_runs).values(
                target_date=target_date, started_at=datetime.now(UTC), status="running"))
            return _pk(res)

    def finish_run(self, run_id: int, status: str, races_seen: int, races_ok: int,
                   errors: list[dict[str, str]]) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(db.collection_runs).where(db.collection_runs.c.run_id == run_id)
                         .values(finished_at=datetime.now(UTC), status=status,
                                 races_seen=races_seen, races_ok=races_ok, errors_json=errors))

    def day_completed(self, target_date: date) -> bool:
        """Vrai si une collecte réussie a eu lieu après la fin de la journée cible."""
        with self.engine.connect() as conn:
            rows = conn.execute(select(db.collection_runs.c.started_at).where(and_(
                db.collection_runs.c.target_date == target_date,
                db.collection_runs.c.status == "success"))).all()
        end_of_day = datetime.combine(target_date, datetime.max.time(), tzinfo=UTC)
        return any(r.started_at > end_of_day for r in rows)

    # ---------------------------------------------------------------- référentiels
    @staticmethod
    def _get_or_create(conn: Connection, table: Table, pk: str, key: dict[str, Any],
                       extra: dict[str, Any]) -> int:
        cond = and_(*[table.c[k] == v if v is not None else table.c[k].is_(None)
                      for k, v in key.items()])
        row = conn.execute(select(table.c[pk]).where(cond)).first()
        if row:
            return int(row[0])
        res = conn.execute(insert(table).values(**key, **extra))
        return _pk(res)

    def _person(self, conn: Connection, table: Table, pk: str, name: str | None) -> int | None:
        canon = canonical_name(name)
        if canon is None or name is None:
            return None
        return self._get_or_create(conn, table, pk, {"canonical_name": canon},
                                   {"display_name": name})

    def _racecourse(self, conn: Connection, race: RaceRecord) -> int:
        return self._get_or_create(conn, db.racecourses, "racecourse_id",
                                   {"code": race.racecourse_code},
                                   {"name": race.racecourse_name, "country": race.country})

    def upsert_race(self, race: RaceRecord, observed_at: datetime, source_key: str,
                    payload_id: int | None) -> int:
        with self.engine.begin() as conn:
            rc_id = self._racecourse(conn, race)
            values = {k: v for k, v in asdict(race).items()
                      if k in db.races.c and k not in ("race_date", "reunion_num", "course_num")}
            values.update(racecourse_id=rc_id, source_key=source_key,
                          last_observed_at=observed_at, payload_id=payload_id)
            key = and_(db.races.c.race_date == race.race_date,
                       db.races.c.reunion_num == race.reunion_num,
                       db.races.c.course_num == race.course_num)
            row = conn.execute(select(db.races.c.race_id).where(key)).first()
            if row:
                conn.execute(update(db.races).where(key).values(**values))
                race_id = int(row[0])
            else:
                res = conn.execute(insert(db.races).values(
                    race_date=race.race_date, reunion_num=race.reunion_num,
                    course_num=race.course_num, first_observed_at=observed_at, **values))
                race_id = _pk(res)
            if race.going_label or race.penetrometer:
                conn.execute(insert(db.racecourse_conditions).values(
                    race_id=race_id, going_label=race.going_label,
                    penetrometer=race.penetrometer, observed_at=observed_at))
            return race_id

    # ---------------------------------------------------------------- instantanés
    def save_snapshot(self, race_id: int, snap: RaceSnapshot, payload_id: int | None,
                      payouts_payload_id: int | None = None) -> None:
        after = snap.observed_after_start
        with self.engine.begin() as conn:
            horse_ids: dict[int, int] = {}
            for e in snap.entries:
                horse_id = self._get_or_create(
                    conn, db.horses, "horse_id",
                    {"canonical_name": e.horse_canonical, "birth_year": e.birth_year},
                    {"display_name": e.horse_name, "breed": e.breed, "sex_last": e.sex})
                horse_ids[e.num_pmu] = horse_id
                h = _entry_hash(e)
                last = conn.execute(
                    select(db.race_entries.c.content_hash)
                    .where(and_(db.race_entries.c.race_id == race_id,
                                db.race_entries.c.num_pmu == e.num_pmu))
                    .order_by(db.race_entries.c.observed_at.desc()).limit(1)).first()
                if last and last[0] == h:
                    continue  # aucune modification depuis la dernière observation
                conn.execute(insert(db.race_entries).values(
                    race_id=race_id, num_pmu=e.num_pmu, horse_id=horse_id,
                    jockey_id=self._person(conn, db.jockeys, "jockey_id", e.rider_name),
                    trainer_id=self._person(conn, db.trainers, "trainer_id", e.trainer_name),
                    owner_id=self._person(conn, db.owners, "owner_id", e.owner_name),
                    age=e.age, sex=e.sex, draw=e.draw, weight_kg=e.weight_kg,
                    handicap_distance_m=e.handicap_distance_m, blinkers=e.blinkers,
                    shoes=e.shoes, status=e.status, form_string=e.form_string,
                    career_starts=e.career_starts, career_wins=e.career_wins,
                    career_places=e.career_places, career_earnings=e.career_earnings,
                    observed_at=snap.observed_at, observed_after_start=after,
                    content_hash=h, payload_id=payload_id))

            if after:
                for r in snap.results:
                    if r.num_pmu not in horse_ids:
                        continue
                    exists = conn.execute(select(db.race_results.c.result_id).where(and_(
                        db.race_results.c.race_id == race_id,
                        db.race_results.c.num_pmu == r.num_pmu))).first()
                    values = {"horse_id": horse_ids[r.num_pmu],
                              "finish_position": r.finish_position, "incident": r.incident,
                              "time_raw": r.time_raw, "reduction_km_raw": r.reduction_km_raw,
                              "observed_at": snap.observed_at, "payload_id": payload_id}
                    if exists:
                        conn.execute(update(db.race_results)
                                     .where(db.race_results.c.result_id == exists[0])
                                     .values(**values))
                    else:
                        conn.execute(insert(db.race_results).values(
                            race_id=race_id, num_pmu=r.num_pmu, **values))

            for o in snap.odds:
                last_odds = conn.execute(
                    select(db.odds_snapshots.c.odds, db.odds_snapshots.c.published_at)
                    .where(and_(db.odds_snapshots.c.race_id == race_id,
                                db.odds_snapshots.c.num_pmu == o.num_pmu,
                                db.odds_snapshots.c.odds_kind == o.odds_kind))
                    .order_by(db.odds_snapshots.c.observed_at.desc()).limit(1)).first()
                if last_odds and last_odds.odds == o.odds \
                        and last_odds.published_at == o.published_at:
                    continue
                conn.execute(insert(db.odds_snapshots).values(
                    race_id=race_id, num_pmu=o.num_pmu, bet_type=o.bet_type,
                    odds_kind=o.odds_kind, odds=o.odds, published_at=o.published_at,
                    observed_at=snap.observed_at, payload_id=payload_id))

            for p in snap.payouts:
                exists = conn.execute(select(db.payouts.c.payout_id).where(and_(
                    db.payouts.c.race_id == race_id, db.payouts.c.bet_type == p.bet_type,
                    db.payouts.c.combination == p.combination))).first()
                if not exists:
                    conn.execute(insert(db.payouts).values(
                        race_id=race_id, bet_type=p.bet_type, combination=p.combination,
                        payout_per_eur=p.payout_per_eur, observed_at=snap.observed_at,
                        payload_id=payouts_payload_id))

    def save_quality_report(self, report: QualityReport, scope: str = "race") -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(db.data_quality_reports).values(
                scope=scope, scope_ref=report.scope_ref, run_at=datetime.now(UTC),
                status=report.status.value, checks_json=report.to_json()))

    # ---------------------------------------------------------------- lectures
    def _df(self, stmt: Any) -> pd.DataFrame:
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            return pd.DataFrame(result.mappings().all(), columns=list(result.keys()))

    def counts(self) -> dict[str, int]:
        tables = {"races": db.races, "horses": db.horses, "entries": db.race_entries,
                  "results": db.race_results, "jockeys": db.jockeys, "trainers": db.trainers,
                  "odds": db.odds_snapshots}
        with self.engine.connect() as conn:
            return {k: int(conn.execute(select(func.count()).select_from(t)).scalar_one())
                    for k, t in tables.items()}

    def last_update(self) -> datetime | None:
        with self.engine.connect() as conn:
            value = conn.execute(select(func.max(db.raw_payloads.c.fetched_at))).scalar()
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def source_statuses(self) -> pd.DataFrame:
        return self._df(select(db.data_sources))

    def recent_runs(self, limit: int = 20) -> pd.DataFrame:
        return self._df(select(db.collection_runs)
                        .order_by(db.collection_runs.c.started_at.desc()).limit(limit))

    def available_dates(self) -> list[date]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(db.races.c.race_date).distinct()
                                .order_by(db.races.c.race_date.desc())).all()
        return [r[0] for r in rows]

    def races_on(self, day: date) -> pd.DataFrame:
        r, rc = db.races, db.racecourses
        latest_q = (select(db.data_quality_reports.c.scope_ref,
                           func.max(db.data_quality_reports.c.report_id).label("rid"))
                    .where(db.data_quality_reports.c.scope == "race")
                    .group_by(db.data_quality_reports.c.scope_ref).subquery())
        stmt = (select(r.c.race_id, r.c.race_date, r.c.reunion_num, r.c.course_num,
                       rc.c.name.label("racecourse"), r.c.label, r.c.discipline,
                       r.c.distance_m, r.c.declared_runners, r.c.scheduled_start_utc,
                       r.c.status, r.c.result_final)
                .join(rc, rc.c.racecourse_id == r.c.racecourse_id, isouter=True)
                .where(r.c.race_date == day)
                .order_by(r.c.reunion_num, r.c.course_num))
        df = self._df(stmt)
        if df.empty:
            return df
        q = self._df(select(db.data_quality_reports.c.scope_ref,
                            db.data_quality_reports.c.status.label("data_status"))
                     .join(latest_q, latest_q.c.rid == db.data_quality_reports.c.report_id))
        df["race_key"] = [f"{d.isoformat()}-R{a}C{b}" for d, a, b in
                          zip(df.race_date, df.reunion_num, df.course_num, strict=True)]
        return df.merge(q, how="left", left_on="race_key", right_on="scope_ref").drop(
            columns=["scope_ref"])

    def entries_as_of(self, race_id: int, as_of: datetime | None = None) -> pd.DataFrame:
        """Dernière version connue de chaque partant à l'instant `as_of` (anti-fuite)."""
        e, h, j, t = db.race_entries, db.horses, db.jockeys, db.trainers
        stmt = (select(e, h.c.display_name.label("horse"), j.c.display_name.label("rider"),
                       t.c.display_name.label("trainer"))
                .join(h, h.c.horse_id == e.c.horse_id)
                .join(j, j.c.jockey_id == e.c.jockey_id, isouter=True)
                .join(t, t.c.trainer_id == e.c.trainer_id, isouter=True)
                .where(e.c.race_id == race_id))
        if as_of is not None:
            stmt = stmt.where(e.c.observed_at <= as_of)
        df = self._df(stmt)
        if df.empty:
            return df
        # drop_duplicates et non groupby().last() : last() combinerait des colonnes de
        # versions différentes en sautant les valeurs nulles.
        df = df.sort_values("observed_at").drop_duplicates("num_pmu", keep="last")
        return df.sort_values("num_pmu").reset_index(drop=True)

    def odds_as_of(self, race_id: int, as_of: datetime,
                   kinds: Sequence[str] = ("reference", "live")) -> pd.DataFrame:
        """Cotes réellement connues à `as_of`. Les cotes `post_start` sont exclues."""
        o = db.odds_snapshots
        kinds = [k for k in kinds if k != "post_start"]
        df = self._df(select(o).where(and_(o.c.race_id == race_id, o.c.observed_at <= as_of,
                                           o.c.odds_kind.in_(kinds))))
        if df.empty:
            return df
        return (df.sort_values("observed_at")
                .drop_duplicates(["num_pmu", "odds_kind"], keep="last")
                .reset_index(drop=True))

    def results_for(self, race_id: int) -> pd.DataFrame:
        return self._df(select(db.race_results).where(db.race_results.c.race_id == race_id)
                        .order_by(db.race_results.c.finish_position))

    def quality_for(self, race_key: str) -> dict[str, Any] | None:
        q = db.data_quality_reports
        with self.engine.connect() as conn:
            row = conn.execute(select(q).where(and_(q.c.scope == "race", q.c.scope_ref == race_key))
                               .order_by(q.c.report_id.desc()).limit(1)).mappings().first()
        return dict(row) if row else None

    def search_horses(self, text: str, limit: int = 50) -> pd.DataFrame:
        canon = canonical_name(text) or ""
        h = db.horses
        return self._df(select(h).where(h.c.canonical_name.contains(canon, autoescape=True))
                        .order_by(h.c.canonical_name).limit(limit))

    def horse_history(self, horse_id: int) -> pd.DataFrame:
        """Historique des courses d'un cheval (dernière version de chaque engagement)."""
        e, r, rc, res, j, t = (db.race_entries, db.races, db.racecourses, db.race_results,
                               db.jockeys, db.trainers)
        stmt = (select(r.c.race_id, r.c.race_date, rc.c.name.label("racecourse"),
                       r.c.discipline, r.c.distance_m, r.c.label, e.c.num_pmu, e.c.status,
                       e.c.draw, e.c.weight_kg, e.c.observed_at,
                       j.c.display_name.label("rider"), t.c.display_name.label("trainer"),
                       res.c.finish_position, res.c.incident)
                .join(r, r.c.race_id == e.c.race_id)
                .join(rc, rc.c.racecourse_id == r.c.racecourse_id, isouter=True)
                .join(j, j.c.jockey_id == e.c.jockey_id, isouter=True)
                .join(t, t.c.trainer_id == e.c.trainer_id, isouter=True)
                .join(res, and_(res.c.race_id == e.c.race_id, res.c.num_pmu == e.c.num_pmu),
                      isouter=True)
                .where(e.c.horse_id == horse_id))
        df = self._df(stmt)
        if df.empty:
            return df
        df = df.sort_values("observed_at").drop_duplicates("race_id", keep="last")
        return df.sort_values("race_date", ascending=False).reset_index(drop=True)
