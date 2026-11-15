"""Rattrape les origines (père, mère, père de la mère) et la valeur handicap officielle
des partants déjà enregistrés (champs collectés depuis le 15/11/2026).

Usage : python scripts/backfill_pedigree.py --start 2024-11-01 --end 2026-11-15
Reprise automatique : relancer la même plage ne relit que les courses incomplètes.
"""

import argparse
import sys
from datetime import date

import _bootstrap  # noqa: F401
from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.ingestion import backfill_pedigree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    pmu = ctx.collectors["pmu"]
    if not isinstance(pmu, PmuCollector):
        raise SystemExit("Connecteur PMU mal configuré.")
    done, absent, errors = backfill_pedigree(pmu, ctx.repo, args.start, args.end)
    print(f"{done} courses complétées, {absent} absentes chez la source, "
          f"{len(errors)} erreur(s)")
    for e in errors[:20]:
        print(f"  - {e}")
    # quelques erreurs réseau isolées n'empêchent pas la reprise suivante
    return 0 if len(errors) <= max(5, done // 100) else 1


if __name__ == "__main__":
    sys.exit(main())
