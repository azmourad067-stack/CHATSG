"""Expérience « quinte_plus » : nombre de combinaisons contre gain au Quinté+.

Usage : python scripts/experiment_quinte_plus.py
Critère fixé AVANT de voir les résultats : pour chaque formule du modèle, N choisi sur la
1re moitié (meilleur ROI), succès si l'IC 95 % du ROI de la 2nde moitié est > 0.
Lecture seule.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_quinte_plus

RANK_LABELS = {"ordre": "ordre", "desordre": "désordre", "bonus4": "bonus 4",
               "bonus3": "bonus 3", "rien": "rien"}


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def main() -> int:
    configure_logging()
    rep = run_quinte_plus(build_context().repo.engine)
    print(f"Quintés : {rep['n_quinte']} en base, {rep['n_quinte_predicted']} avec prédictions, "
          f"{rep['n_quinte_settled']} avec rapports complets")
    print("Mise de base 2 € par combinaison ; chaque combinaison paie son meilleur rang.")
    for name, fam in rep["families"].items():
        print(f"\n[{name}]")
        for n, r in fam.items():
            tot = r["periods"].get("total", {})
            if not tot.get("n_races"):
                print(f"  N={n} —")
                continue
            first = r["periods"]["1re moitié"]
            second = r["periods"]["2nde moitié"]
            ranks = ", ".join(f"{RANK_LABELS[k]} {100 * v:.0f} %" for k, v in r["ranks"].items())
            print(f"  N={n:2d} : {r['n_tickets']:3d} combinaisons = "
                  f"{2 * r['n_tickets']:3d} € par quinté | total {tot['profit']:+.0f} € sur "
                  f"{tot['staked']:.0f} € ({_pct(tot['roi'])}, IC [{_pct(tot['roi_ci'][0])} ; "
                  f"{_pct(tot['roi_ci'][1])}])")
            print(f"         1re moitié {_pct(first['roi'])} ; 2nde moitié {_pct(second['roi'])} "
                  f"(IC [{_pct(second['roi_ci'][0])} ; {_pct(second['roi_ci'][1])}])")
            print(f"         meilleur rang atteint : {ranks}")
            print(f"         gain médian quand ça paie {tot['median_hit_return']:.0f} €, "
                  f"meilleur {tot['best_return']:.0f} €, pire série "
                  f"{tot['longest_losing_streak']} quintés, creux max "
                  f"{tot['max_drawdown']:.0f} €")
    for name, n in rep["chosen"].items():
        print(f"\nChoisi sur la 1re moitié pour « {name} » : N={n}")
    verdict = ("SUCCÈS : une formule est significativement rentable sur la 2nde moitié"
               if rep["success"] else "ÉCHEC : aucune formule significativement rentable")
    print(f"\nCritère fixé à l'avance (N choisi sur la 1re moitié, 2nde moitié IC 95 % du "
          f"ROI > 0) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
