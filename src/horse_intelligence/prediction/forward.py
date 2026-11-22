"""Prédictions enregistrées AVANT le départ (test en conditions réelles).

Chaque prédiction est figée en base avec l'instant de calcul, la version du modèle et une
empreinte des données d'entrée. Comparée ensuite au résultat, elle fournit la seule
évaluation totalement exempte de fuite : rien de postérieur n'a pu l'influencer.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import Engine, and_, insert, select

from horse_intelligence.data.storage import database as db
from horse_intelligence.intelligence.feature_engineering import FORM_FEATURES, MARKET_FEATURES
from horse_intelligence.prediction.race_predictor import (
    build_frame,
    load_models,
    models_use_source_aggregates,
    predict_race,
)


def _input_hash(race: pd.DataFrame) -> str:
    cols = ["num_pmu", *FORM_FEATURES, *MARKET_FEATURES]
    rows = race[[c for c in cols if c in race.columns]].sort_values("num_pmu")
    body = json.dumps(rows.to_dict("records"), sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def record_upcoming(engine: Engine, window: timedelta, now: datetime | None = None,
                    models: dict[str, Any] | None = None,
                    frames: dict[object, pd.DataFrame] | None = None) -> int:
    """Enregistre une prédiction pour chaque course partant dans la fenêtre. `models` et
    `frames` (par jour) peuvent être fournis déjà construits pour éviter de les refaire."""
    now = now or datetime.now(UTC)
    r = db.races
    with engine.connect() as conn:
        upcoming = conn.execute(select(r.c.race_id, r.c.race_date).where(and_(
            r.c.scheduled_start_utc > now, r.c.scheduled_start_utc <= now + window))).all()
    if not upcoming:
        return 0
    models = models if models is not None else load_models(engine)
    if not models:
        return 0
    frames = frames if frames is not None else {}
    saved = 0
    for race_id, race_date in upcoming:
        if race_date not in frames:
            frames[race_date] = build_frame(engine, race_date, models_use_source_aggregates(models))
        frame = frames[race_date]
        pred = predict_race(engine, int(race_id), frame=frame, models=models)
        if pred.table.empty or pred.primary is None:
            continue
        race = frame[frame["race_id"] == race_id]
        with engine.begin() as conn:
            res = conn.execute(insert(db.predictions).values(
                race_id=int(race_id), model_id=int(pred.models[pred.primary]["model_id"]),
                decision_time=now, created_at=datetime.now(UTC), data_status=pred.status,
                input_snapshot_hash=_input_hash(race)))
            pid = int(res.inserted_primary_key[0]) if res.inserted_primary_key else 0
            t = pred.table
            conn.execute(insert(db.prediction_entries), [{
                "prediction_id": pid, "num_pmu": int(row["num_pmu"]),
                "p_win": float(row["p_win"]), "p_place": _f(row.get("p_place")),
                "p_win_lo": _f(row.get("p_low")), "p_win_hi": _f(row.get("p_high")),
                "rank": int(row["rank"]), "model_disagreement": _f(row.get("disagreement")),
                "mkt_prob": _f(row.get("mkt_prob")), "mkt_odds": _f(row.get("mkt_odds")),
            } for row in t.to_dict("records")])
        saved += 1
    return saved


def _f(v: object) -> float | None:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) else x


def precompute_day(engine: Engine, day: date, now: datetime | None = None,
                   models: dict[str, Any] | None = None,
                   frames: dict[object, pd.DataFrame] | None = None) -> dict[int, Any]:
    """Calcule et range les probabilités de toutes les courses de `day` pour l'interface,
    s'il reste au moins une course à courir. Renvoie les prédictions (vide sinon)."""
    from horse_intelligence.prediction import day_store
    from horse_intelligence.prediction.race_predictor import predict_day

    now = now or datetime.now(UTC)
    r = db.races
    with engine.connect() as conn:
        races = conn.execute(select(r.c.race_id, r.c.race_date, r.c.scheduled_start_utc)
                             .where(r.c.race_date == day)).all()
    if not races or not any(x.scheduled_start_utc is not None
                            and _utc(x.scheduled_start_utc) > now for x in races):
        return {}
    models = models if models is not None else load_models(engine)
    if not models:
        return {}
    # clé lue AVANT le calcul : une donnée arrivée pendant le calcul la rend périmée
    key = day_store.data_key(engine)
    frames = frames if frames is not None else {}
    if day not in frames:
        frames[day] = build_frame(engine, day, models_use_source_aggregates(models))
    ids = [int(x.race_id) for x in races]
    preds = predict_day(engine, ids, frames[day], models)
    day_store.store(engine, preds, {int(x.race_id): x.race_date for x in races}, key, now)
    return preds


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)
