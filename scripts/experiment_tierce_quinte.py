"""Expérience « tierce_quinte » : 20 combinaisons de tiercé par course du quinté.

Usage : python scripts/experiment_tierce_quinte.py
Critère fixé AVANT de voir les résultats : une stratégie du modèle est rentable sur la 2nde
moitié, IC 95 % du ROI entièrement > 0. Lecture seule.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_tierce_quinte


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def main() -> int:
    configure_logging()
    rep = run_tierce_quinte(build_context().repo.engine)
    print(f"Quintés : {rep['n_quinte']} en base, {rep['n_quinte_predicted']} avec prédictions, "
          f"{rep['n_quinte_settled']} avec rapports tiercé ordre ET désordre "
          f"(forces des trios : {rep['strengths']})")
    print("20 combinaisons à 1 € = 20 € par quinté")
    for name, periods in rep["results"].items():
        print(f"\n[{name}]")
        for p, r in periods.items():
            if not r.get("n_races"):
                print(f"  {p:12s} —")
                continue
            print(f"  {p:12s} {r['n_races']} quintés, misé {r['staked']:.0f} €, "
                  f"résultat {r['profit']:+.0f} € ({_pct(r['roi'])}, IC [{_pct(r['roi_ci'][0])} ; "
                  f"{_pct(r['roi_ci'][1])}]) ; trio trouvé {100 * r['hit_rate']:.0f} % "
                  f"(dont ordre {100 * r['exact_rate']:.0f} %), rapport médian quand trouvé "
                  f"{r['median_hit_return']:.0f} €, meilleur {r['best_return']:.0f} €, "
                  f"pire série {r['longest_losing_streak']} quintés, creux max "
                  f"{r['max_drawdown']:.0f} €")
    verdict = ("SUCCÈS : une stratégie est significativement rentable sur la 2nde moitié"
               if rep["success"] else "ÉCHEC : aucune stratégie significativement rentable")
    print(f"\nCritère fixé à l'avance (2nde moitié, IC 95 % du ROI > 0) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
