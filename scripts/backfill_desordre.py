"""Rattrape les rapports « désordre » (tiercé, quarté+, quinté+) des courses du quinté.

Usage : python scripts/backfill_desordre.py
Idempotent : seules les courses sans rapport TIERCE_DESORDRE sont relues.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.ingestion import backfill_quinte_desordre


def main() -> int:
    configure_logging()
    ctx = build_context()
    pmu = ctx.collectors["pmu"]
    if not isinstance(pmu, PmuCollector):
        raise SystemExit("Connecteur PMU mal configuré.")
    done, absent, errors = backfill_quinte_desordre(pmu, ctx.repo)
    print(f"Rapports désordre : {done} quintés complétés, {absent} absents chez la source, "
          f"{len(errors)} erreur(s)")
    for e in errors[:20]:
        print(f"  - {e}")
    return 0 if len(errors) <= max(5, done // 50) else 1


if __name__ == "__main__":
    sys.exit(main())
