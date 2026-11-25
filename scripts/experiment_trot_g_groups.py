"""Expérience « trot_g_groups » : règle trot G jouée sur plusieurs chevaux en simple
gagnant (2 ou 3 rangs du modèle, 1 € par cheval ou dutching).

Usage : python scripts/experiment_trot_g_groups.py
Grille fixée AVANT de voir les résultats (evaluation/trot_g_groups.py). Lecture seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_trot_g_groups


def _pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:+.1f} %"


def _line(r: dict[str, Any]) -> str:
    if not r.get("bets"):
        return "aucun pari"
    return (f"{r['races']} courses, mise {r['bets']:.0f} €, {r['profit']:+.0f} € "
            f"({_pct(r['roi'])}, IC [{_pct(r['roi_ci'][0])} ; {_pct(r['roi_ci'][1])}]), "
            f"gain sur {100 * r['hit_rate']:.0f} % des courses, pire baisse "
            f"{r['max_drawdown']:.0f} €")


def main() -> int:
    configure_logging()
    rep = run_trot_g_groups(build_context().repo.engine)
    if not rep.get("n_races"):
        print("Aucune course de trot G")
        return 0
    print(f"{rep['n_races']} courses de trot G ouvertes au pari en ligne, "
          f"{rep['period'][0]} → {rep['period'][1]}")
    print("\n[Chaque rang du modèle joué seul, 1 € en simple gagnant, deux ans]")
    for r in rep["by_rank"]:
        print(f"  {r['rank']}e du modèle : {r['bets']} paris, ROI {_pct(r['roi'])}, "
              f"{100 * r['hit_rate']:.0f} % de gagnants, cote médiane {r['odds']:.1f}")
    print(f"\n[Groupes de chevaux ; IC sur deux ans à {100 * (1 - rep['alpha']):.2f} % "
          "(14 essais)]")
    for s in rep["strategies"]:
        print(f"  • {s['label']}")
        print(f"      deux ans          {_line(s['all'])}")
        print(f"      6 derniers mois   {_line(s['last6'])}")
    verdict = ("SUCCÈS : un groupe est significativement gagnant sur deux ans"
               if rep["success"] else "ÉCHEC : aucun groupe n'est significativement gagnant")
    print(f"\nCritère fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
