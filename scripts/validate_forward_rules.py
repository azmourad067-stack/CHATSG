"""Bilan des règles figées le 21/11/2026 sur les courses postérieures (prédictions
enregistrées AVANT le départ, rapports définitifs Internet).

Usage : python scripts/validate_forward_rules.py
Critère fixé à l'avance : IC 95 % du ROI entièrement > 0 sur au moins 600 paris. Lecture
seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.evaluation.forward_rules import (
    D4_AUTOSTART,
    MIN_BETS_VERDICT,
    SMALL_VALUE,
    SMALL_VALUE_EARLY,
    VALUE_46,
    VALUE_46_EARLY,
    forward_bets,
    forward_d4,
    forward_trot_g,
    verdict,
)
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def _print(name: str, v: dict[str, Any]) -> None:
    print(f"\n[{name}]")
    if not v.get("bets"):
        print(f"  {v['status']}")
        return
    print(f"  {v['period'][0]} → {v['period'][1]} : {v['bets']} paris, résultat "
          f"{v['profit']:+.0f} € ({_pct(v['roi'])}, IC 95 % [{_pct(v['roi_ci'][0])} ; "
          f"{_pct(v['roi_ci'][1])}]), {100 * v['hit_rate']:.0f} % de gagnants")
    print(f"  Bilan : {v['status']}")


def main() -> int:
    configure_logging()
    engine = build_context().repo.engine
    print(f"Règles jugées à partir du {VALUE_46.first_day} (gel le {VALUE_46.frozen_on}) ; "
          f"verdict à partir de {MIN_BETS_VERDICT} paris")
    _print(f"{VALUE_46.name} (toutes réunions ouvertes aux paris en ligne)",
           verdict(forward_bets(engine)))
    _print(f"{VALUE_46_EARLY.name} (figée le {VALUE_46_EARLY.frozen_on}, jugée à partir du "
           f"{VALUE_46_EARLY.first_day} ; règle des alertes SMS)",
           verdict(forward_bets(engine, VALUE_46_EARLY)))
    _print(f"{SMALL_VALUE.name} (figée le {SMALL_VALUE.frozen_on}, jugée à partir du "
           f"{SMALL_VALUE.first_day} ; décision 13 à 28 min avant le départ)",
           verdict(forward_bets(engine, SMALL_VALUE)))
    _print(f"{SMALL_VALUE_EARLY.name} (à comparer à la ligne précédente)",
           verdict(forward_bets(engine, SMALL_VALUE_EARLY)))
    _print(f"{D4_AUTOSTART.name} (figée le {D4_AUTOSTART.frozen_on}, jugée à partir du "
           f"{D4_AUTOSTART.first_day})", verdict(forward_d4(engine)))
    _print("3e du modèle, gagnant, trot G (toutes réunions ouvertes aux paris en ligne)",
           verdict(forward_trot_g(engine)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
