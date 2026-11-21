"""Expérience « grand_search » : simples et combinés, découverte / validation / test final.

Usage : python scripts/experiment_grand_search.py
Protocole fixé AVANT de voir les résultats (voir evaluation/grand_search.py). Lecture seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_grand_search


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def _line(r: dict[str, Any]) -> str:
    if not r.get("bets"):
        return "aucun pari"
    return (f"{r['bets']:.0f} € misés (~{r['stake_per_year']:.0f} €/an), résultat "
            f"{r['profit']:+.0f} € ({_pct(r['roi'])}, IC [{_pct(r['roi_ci'][0])} ; "
            f"{_pct(r['roi_ci'][1])}]), gain sur {100 * r['hit_rate']:.0f} % des courses, "
            f"creux max {r['max_drawdown']:.0f} €")


def _block(r: dict[str, Any]) -> None:
    print(f"  • {r['label']}")
    for k, name in (("A", "A découverte"), ("B", "B validation"), ("C", "C test final")):
        print(f"      {name:14s} {_line(r[k])}")


def main() -> int:
    configure_logging()
    rep = run_grand_search(build_context().repo.engine)
    if not rep.get("n_races"):
        print("Aucune course.")
        return 0
    print(f"{rep['n_races']} courses, {rep['period'][0]} → {rep['period'][1]} ; "
          f"A avant le {rep['cuts'][0]}, B jusqu'au {rep['cuts'][1]}, C ensuite")
    print(f"{rep['n_strategies']} stratégies ({rep['n_simple']} en simple, "
          f"{rep['n_strategies'] - rep['n_simple']} en combinés), {rep['n_eligible']} assez "
          f"fournies ; mise 1 € par cheval / combinaison, rapport définitif Internet")
    c = rep["chance"]
    print(f"\n[Repère du hasard] parmi {c['n']} stratégies fournies sur les 3 périodes : "
          f"{c['positive_a']} gagnantes en A, {c['positive_b']} en B, {c['positive_c']} en C, "
          f"{c['positive_all']} sur les trois ; ROI moyen en C des {rep['n_discovery']} "
          f"meilleures de A : {_pct(c['roi_c_all_discovered'])}")
    print("\n[Les 5 meilleures de la découverte (A), pour information]")
    for r in rep["top_discovery"]:
        _block(r)
    print(f"\n[{rep['n_validated']} des {rep['n_discovery']} meilleures de A restent "
          f"positives en B ; finalistes (meilleur ROI en B), IC en C à "
          f"{100 * (1 - rep['alpha']):.0f} % corrigé pour {len(rep['finalists'])} essais]")
    for r in rep["finalists"]:
        _block(r)
    verdict = ("SUCCÈS : une stratégie reste significativement gagnante sur la période C"
               if rep["success"] else "ÉCHEC : aucune stratégie ne tient sur la période C")
    print(f"\nCritère fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
