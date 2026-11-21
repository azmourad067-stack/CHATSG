"""Enregistre les prédictions des courses qui partent bientôt (test en conditions réelles)
et précalcule les probabilités de toutes les courses du jour pour l'interface.

Usage : python scripts/predict_upcoming.py [--window-minutes 75]
À lancer après chaque collecte intrajournalière (tâche planifiée).
"""

import argparse
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.prediction.forward import precompute_day, record_upcoming
from horse_intelligence.prediction.race_predictor import load_models
from horse_intelligence.services.context import build_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-minutes", type=int, default=75)
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    engine = ctx.repo.engine
    # modèles et variables du jour construits une seule fois pour les deux calculs
    models = load_models(engine)
    frames: dict[object, pd.DataFrame] = {}
    n = record_upcoming(engine, timedelta(minutes=args.window_minutes), models=models,
                        frames=frames)
    print(f"{n} prédiction(s) enregistrée(s) avant départ")
    try:
        today = datetime.now(ZoneInfo("Europe/Paris")).date()
        k = precompute_day(engine, today, models=models, frames=frames)
        print(f"{k} course(s) du jour précalculée(s) pour l'interface")
    except Exception as exc:  # l'interface recalculera elle-même : jamais bloquant
        print(f"::warning::précalcul du jour en échec : {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
