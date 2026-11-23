"""Expérience « d4_draw » : déferré des 4 pour la 1re fois × numéro de corde, en gagnant.

Usage : python scripts/experiment_d4_draw.py
Protocole fixé AVANT de voir les résultats (evaluation/d4_draw.py). Lecture seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_d4_draw


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
    rep = run_d4_draw(build_context().repo.engine)
    if not rep.get("n_races"):
        print(f"Aucune course (variables absentes : {rep.get('missing_columns')})")
        return 0
    print(f"{rep['n_races']} courses, {rep['period'][0]} → {rep['period'][1]} ; A avant le "
          f"{rep['cuts'][0]}, B jusqu'au {rep['cuts'][1]}, C ensuite")
    print(f"{rep['n_d4']} trotteurs déferrés des 4 pour la 1re fois ; numéro de corde publié "
          f"pour {100 * rep['draw_published_share']:.0f} % d'entre eux (sinon numéro PMU)")
    if rep["missing_columns"]:
        print(f"Variables absentes : {', '.join(rep['missing_columns'])}")
    for group, title in (("d4", "Déferrés des 4 pour la 1re fois"),
                         ("autres", "Repère : autres trotteurs")):
        print(f"\n[{title}, toutes cotes, 2 ans ; q = significativité corrigée]")
        for a in rep["table"]:
            if a["group"] == group:
                print(f"  {a['label']:40s} {a['bets']:6d} paris  ROI {_pct(a['roi'])}  "
                      f"(A {_pct(a['roi_A'])}, C {_pct(a['roi_C'])})  gagnants "
                      f"{100 * a['hit_rate']:.0f} %  q = {a['q']:.2f}")
    c = rep["chance"]
    print(f"\n[Repère du hasard] {rep['n_strategies']} combinaisons, {c['n']} assez fournies : "
          f"{c['positive_all']} gagnantes sur les trois périodes ; ROI moyen en C des 10 "
          f"meilleures de A : {_pct(c['roi_c_discovered'])}")
    print("\n[Les 5 meilleures de A]")
    for r in rep["top_discovery"]:
        _block(r)
    print(f"\n[{rep['n_validated']} des 10 meilleures de A restent positives en B ; "
          f"finalistes, IC en C à {100 * (1 - rep['alpha']):.1f} %]")
    for r in rep["finalists"]:
        _block(r)
    verdict = ("SUCCÈS : une combinaison reste significativement gagnante sur la période C"
               if rep["success"] else "ÉCHEC : aucune combinaison ne tient sur la période C")
    print(f"\nCritère fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
