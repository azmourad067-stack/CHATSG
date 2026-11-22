"""Expérience « angles » : signaux de turfiste en simple gagnant (déferré, œillères, forme
du driver, catégorie, repos, spécialiste…), croisés avec la cote et la discipline.

Usage : python scripts/experiment_angles.py
Protocole fixé AVANT de voir les résultats (evaluation/angles.py). Lecture seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_angles


def _pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:+.1f} %"


def _line(r: dict[str, Any]) -> str:
    if not r.get("bets"):
        return "aucun pari"
    return (f"{r['bets']:.0f} paris (~{r['stake_per_year']:.0f}/an), {r['profit']:+.0f} € "
            f"({_pct(r['roi'])}, IC [{_pct(r['roi_ci'][0])} ; {_pct(r['roi_ci'][1])}]), "
            f"gain sur {100 * r['hit_rate']:.0f} % des courses")


def _block(r: dict[str, Any]) -> None:
    print(f"  • {r['label']}")
    for k, name in (("A", "A découverte"), ("B", "B validation"), ("C", "C test final")):
        print(f"      {name:14s} {_line(r[k])}")


def main() -> int:
    configure_logging()
    rep = run_angles(build_context().repo.engine)
    if not rep.get("n_races"):
        print("Aucune course.")
        return 0
    print(f"{rep['n_races']} courses, {rep['period'][0]} → {rep['period'][1]} ; A avant le "
          f"{rep['cuts'][0]}, B jusqu'au {rep['cuts'][1]}, C ensuite")
    if rep["missing_columns"]:
        print(f"Variables absentes : {', '.join(rep['missing_columns'])}")
    print(f"Repère : tous les partants joués en gagnant → {_line(rep['baseline'])}")
    print("\n[Chaque angle seul, toutes cotes et disciplines, 2 ans ; q = significativité "
          "corrigée du nombre d'essais]")
    for a in rep["angles"]:
        print(f"  {a['angle']:44s} {a['bets']:6d} paris  ROI {_pct(a['roi'])}  "
              f"(A {_pct(a['roi_A'])}, C {_pct(a['roi_C'])})  gagnants "
              f"{100 * a['hit_rate']:.0f} %  q = {a['q']:.2f}")
    c = rep["chance"]
    print(f"\n[Repère du hasard] {rep['n_strategies']} combinaisons angle × cote × discipline, "
          f"{c['n']} assez fournies : {c['positive_a']} gagnantes en A, {c['positive_b']} en "
          f"B, {c['positive_c']} en C, {c['positive_all']} sur les trois ; ROI moyen en C des "
          f"20 meilleures de A : {_pct(c['roi_c_discovered'])}")
    print("\n[Les 5 meilleures de A]")
    for r in rep["top_discovery"]:
        _block(r)
    print(f"\n[{rep['n_validated']} des 20 meilleures de A restent positives en B ; "
          f"finalistes, IC en C à {100 * (1 - rep['alpha']):.0f} %]")
    for r in rep["finalists"]:
        _block(r)
    verdict = ("SUCCÈS : un angle reste significativement gagnant sur la période C"
               if rep["success"] else "ÉCHEC : aucun angle ne tient sur la période C")
    print(f"\nCritère fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
