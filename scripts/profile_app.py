"""Mesure des temps de calcul de la page « Sélection du jour » sur la vraie base.

Usage : python scripts/profile_app.py [--day AAAA-MM-JJ]
Lecture seule : rien n'est écrit en base.
"""

import argparse
import sys
import time
from datetime import date

from sqlalchemy import select

import _bootstrap  # noqa: F401
from horse_intelligence.data.storage import database as db
from horse_intelligence.intelligence.feature_engineering import build_dataset, load_raw
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.prediction.race_predictor import (
    load_models,
    models_use_source_aggregates,
    predict_day,
    predict_race,
)
from horse_intelligence.services.context import build_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    configure_logging()
    repo = build_context().repo
    engine = repo.engine
    day = args.day or max(repo.available_dates())
    t = time.perf_counter()

    def lap(label: str) -> None:
        nonlocal t
        now = time.perf_counter()
        print(f"{label:55s} {now - t:7.1f} s")
        t = now

    models = load_models(engine)
    lap("chargement des modèles")
    raw = load_raw(engine, until=day)
    lap("lecture de l'historique en base (load_raw)")
    for k, v in raw.items():
        print(f"    {k:12s} {len(v):>9d} lignes")
    frame = build_dataset(raw, use_source_aggregates=models_use_source_aggregates(models)).frame
    lap("calcul des variables (build_dataset)")
    del raw
    with engine.connect() as conn:
        races = conn.execute(select(db.races.c.race_id, db.races.c.audience)
                             .where(db.races.c.race_date == day)).all()
    national = [r.race_id for r in races if r.audience not in ("LOCAL", "REGIONAL")]
    print(f"    {day} : {len(races)} courses, {len(national)} au programme national")
    lap("liste des courses du jour")
    for rid in national:
        predict_race(engine, int(rid), frame=frame, models=models)
    lap(f"predict_race × {len(national)} (complet, comme la page)")
    for rid in national:
        repo.entries_as_of(int(rid))
    lap(f"entries_as_of × {len(national)}")
    predict_day(engine, [int(r) for r in national], frame, models)
    lap(f"predict_day (nouveau : toutes les courses en un calcul, {len(national)})")
    repo.entries_for_races([int(r) for r in national])
    lap("entries_for_races (nouveau : une seule requête)")
    day_frame = frame[frame["race_id"].isin(national)]
    primary = models.get("lgbm_market") or models.get("lgbm_form")
    if primary is not None and primary.lgbm is not None:
        sub = day_frame[day_frame["mkt_prob"].notna()] if primary.lgbm.with_market \
            else day_frame
        primary.lgbm.predict(sub)
        lap(f"probabilités en un seul calcul ({sub['race_id'].nunique()} courses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
