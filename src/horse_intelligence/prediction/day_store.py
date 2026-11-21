"""Prédictions du jour précalculées par la collecte, lues par l'interface.

La collecte (toutes les 15 min) calcule les probabilités de toutes les courses du jour
avec `predict_day` et les range en base avec la CLÉ DE FRAÎCHEUR des données (versions des
modèles + dernière donnée reçue). L'interface ne les utilise que si cette clé est celle du
moment : sinon (nouvelle collecte, nouveau modèle), elle recalcule elle-même. Les chiffres
affichés sont donc toujours ceux d'un calcul fait sur les données les plus récentes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
from sqlalchemy import Engine, delete, func, insert, select

from horse_intelligence.data.storage import database as db
from horse_intelligence.models.registry import latest_model
from horse_intelligence.prediction.race_predictor import DayPrediction

KEY_FAMILIES = ("lgbm_market", "lgbm_form", "market_calibrated", "place_market")
KEEP_DAYS = 3


def model_key(engine: Engine) -> str:
    """Versions des modèles utilisés (même clé pour la collecte et l'interface)."""
    metas = [latest_model(engine, f) for f in KEY_FAMILIES]
    return "|".join(str(m["version"]) if m else "-" for m in metas)


def last_update(engine: Engine) -> datetime | None:
    with engine.connect() as conn:
        value = conn.execute(select(func.max(db.raw_payloads.c.fetched_at))).scalar()
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def data_key(engine: Engine, mkey: str | None = None) -> str:
    """Clé de fraîcheur : un calcul fait avec une autre clé n'est jamais affiché."""
    last = last_update(engine)
    return f"{mkey if mkey is not None else model_key(engine)}|" \
           f"{last.isoformat() if last else '-'}"


def store(engine: Engine, preds: dict[int, DayPrediction], race_dates: dict[int, date],
          key: str, now: datetime | None = None) -> int:
    """Remplace les prédictions stockées de ces courses ; purge les anciens jours."""
    now = now or datetime.now(UTC)
    ids = list(preds)
    races, runners = db.day_prediction_races, db.day_prediction_runners
    race_rows = [{"race_id": rid, "race_date": race_dates[rid], "data_key": key,
                  "primary_model": p.primary, "message": (p.message or "")[:300],
                  "computed_at": now} for rid, p in preds.items()]
    runner_rows = [{"race_id": rid, "num_pmu": int(r["num_pmu"]),
                    "horse_id": None if pd.isna(r["horse_id"]) else int(r["horse_id"]),
                    "p_win": _f(r["p_win"]), "mkt_odds": _f(r["mkt_odds"]),
                    "mkt_prob": _f(r["mkt_prob"]), "rank": _f(r["rank"])}
                   for rid, p in preds.items() if not p.table.empty
                   for r in p.table.to_dict("records")]
    oldest = min(race_dates.values(), default=now.date()) - timedelta(days=KEEP_DAYS)
    with engine.begin() as conn:
        stale = select(races.c.race_id).where(races.c.race_date < oldest)
        conn.execute(delete(runners).where(runners.c.race_id.in_(stale)))
        conn.execute(delete(races).where(races.c.race_date < oldest))
        if ids:
            conn.execute(delete(runners).where(runners.c.race_id.in_(ids)))
            conn.execute(delete(races).where(races.c.race_id.in_(ids)))
        if race_rows:
            conn.execute(insert(races), race_rows)
        if runner_rows:
            conn.execute(insert(runners), runner_rows)
    return len(race_rows)


def load(engine: Engine, race_ids: list[int], key: str) -> dict[int, DayPrediction] | None:
    """Prédictions stockées de TOUTES ces courses avec la clé `key`, sinon None."""
    if not race_ids:
        return {}
    races, runners = db.day_prediction_races, db.day_prediction_runners
    with engine.connect() as conn:
        rows = conn.execute(select(races).where(races.c.race_id.in_(race_ids))).all()
        if len(rows) < len(set(race_ids)) or any(r.data_key != key for r in rows):
            return None
        tab = pd.read_sql(select(runners).where(runners.c.race_id.in_(race_ids)), conn)
    groups = {int(cast(int, rid)): g for rid, g in tab.groupby("race_id")} if not tab.empty \
        else {}
    out: dict[int, DayPrediction] = {}
    for r in rows:
        g = groups.get(int(r.race_id))
        table = pd.DataFrame() if g is None or r.primary_model is None else (
            g[["num_pmu", "horse_id", "mkt_odds", "mkt_prob", "p_win", "rank"]]
            .sort_values(["rank", "num_pmu"]).reset_index(drop=True))
        out[int(r.race_id)] = DayPrediction(int(r.race_id), r.primary_model, table,
                                            r.message or "")
    return out


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) else x
