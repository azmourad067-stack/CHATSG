"""Expérience « value46 » : règle Valeur 4e-6e en gagnant, en placé et les deux.

Usage : python scripts/experiment_value46.py
Descriptif, lecture seule : la règle figée le 21/11/2026 reste en simple gagnant.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_value46_bet_types


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def _line(r: dict[str, Any]) -> str:
    if not r.get("bets"):
        return "aucun pari"
    return (f"{r['bets']:.0f} € misés (~{r['stake_per_year']:.0f} €/an), résultat "
            f"{r['profit']:+.0f} € ({_pct(r['roi'])}, IC [{_pct(r['roi_ci'][0])} ; "
            f"{_pct(r['roi_ci'][1])}]), gain sur {100 * r['hit_rate']:.0f} % des courses, "
            f"creux max {r['max_drawdown']:.0f} €")


def main() -> int:
    configure_logging()
    rep = run_value46_bet_types(build_context().repo.engine)
    if not rep.get("n_races"):
        print("Aucune course.")
        return 0
    print(f"{rep['n_races']} courses, {rep['period'][0]} → {rep['period'][1]} ; A avant le "
          f"{rep['cuts'][0]}, B jusqu'au {rep['cuts'][1]}, C ensuite ; 1 € par cheval et par "
          f"pari, rapports définitifs Internet")
    for name, periods in rep["results"].items():
        print(f"\n[Valeur 4e-6e en {name}]")
        for k, label in (("A", "A (1re année)"), ("B", "B (6 mois)"), ("C", "C (6 derniers mois)"),
                         ("total", "total")):
            print(f"  {label:20s} {_line(periods[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
