"""Expérience « market_bias » : le marché se trompe-t-il davantage dans certains contextes ?

Usage : python scripts/experiment_market_bias.py
Carte du marché par contexte, puis 4 stratégies regroupées fixées AVANT de voir les
résultats (evaluation/market_bias.py). Lecture seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_market_bias


def _pct(x: float) -> str:
    return "   —   " if x != x else f"{100 * x:+6.1f} %"


def _line(r: dict[str, Any]) -> str:
    if not r.get("bets"):
        return "aucun pari"
    return (f"{r['bets']:.0f} paris (~{r['stake_per_year']:.0f}/an), {r['profit']:+.0f} € "
            f"({_pct(r['roi']).strip()}, IC [{_pct(r['roi_ci'][0]).strip()} ; "
            f"{_pct(r['roi_ci'][1]).strip()}]), gain sur {100 * r['hit_rate']:.0f} % des "
            f"courses")


def main() -> int:
    configure_logging()
    rep = run_market_bias(build_context().repo.engine)
    if not rep.get("n_races"):
        print("Aucune course")
        return 0
    print(f"{rep['n_races']} courses, {rep['period'][0]} → {rep['period'][1]} ; A avant le "
          f"{rep['cuts'][0]}, B jusqu'au {rep['cuts'][1]}, C ensuite")
    print("\n[Carte du marché sur deux ans : perte en jouant TOUS les partants ; "
          "apport du modèle > 0 = il prévoit mieux que les cotes]")
    print(f"  {'contexte':44s} {'courses':>7s}  {'gagnant':>8s}  {'placé':>8s}  "
          f"{'cote≤4':>8s} {'4-9':>8s} {'9-20':>8s} {'20-50':>8s}  apport modèle")
    for r in sorted(rep["map"], key=lambda x: -(x["roi_gagnant"]
                                                if x["roi_gagnant"] == x["roi_gagnant"]
                                                else -9)):
        bands = " ".join(_pct(r[f"roi_gagnant_{i}"]) for i in range(4))
        print(f"  {r['context'][:44]:44s} {r['races']:7d}  {_pct(r['roi_gagnant'])} "
              f"{_pct(r['roi_placé'])}  {bands}  {1000 * r['model_gain']:+.2f} ‰")
    print(f"\n[4 stratégies regroupées ; IC en C à {100 * (1 - rep['alpha']):.2f} %]")
    for s in rep["strategies"].values():
        print(f"\n  • {s['label']} ({s['n_cells']} cases possibles)")
        ab, a = s["AB"], s["A"]
        print(f"      cases apprises sur A+B : {ab['n_kept']} ; sur A+B {_line(ab['train'])}")
        print(f"      → C test final      {_line(ab['C'])}")
        print(f"      cases apprises sur A seul : {a['n_kept']} ; sur A {_line(a['train'])}")
        print(f"      → B                 {_line(a['B'])}")
        print(f"      → C                 {_line(a['C'])}")
        for c in ab["top_cells"][:5]:
            print(f"        - {c['label']} ({c['bets']} paris, ROI rétréci "
                  f"{_pct(c['roi_shrunk']).strip()})")
    verdict = ("SUCCÈS : une stratégie regroupée reste significativement gagnante sur C"
               if rep["success"] else "ÉCHEC : aucune stratégie regroupée ne tient sur C")
    print(f"\nCritère fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
