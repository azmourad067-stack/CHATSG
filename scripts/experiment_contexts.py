"""Expérience « contexts » : recherche par contexte de course (hippodrome, réunion,
catégorie, distance, allocation, jour), simple gagnant et placé.

Usage : python scripts/experiment_contexts.py
Protocole fixé AVANT de voir les résultats (evaluation/context_search.py). Lecture seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_context_search


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
    rep = run_context_search(build_context().repo.engine)
    if not rep.get("n_races"):
        print("Aucune course")
        return 0
    print(f"{rep['n_races']} courses, {rep['period'][0]} → {rep['period'][1]} ; A avant le "
          f"{rep['cuts'][0]}, B jusqu'au {rep['cuts'][1]}, C ensuite")
    kinds = ", ".join(f"{k} : {n}" for k, n in rep["context_kinds"].items())
    print(f"{rep['n_contexts']} contextes ({kinds}) × 64 sélections = "
          f"{rep['n_strategies']} stratégies, {rep['n_eligible']} assez fournies")
    for b, r in rep["baseline"].items():
        print(f"Repère, tous les partants en {b} : {_line(r)}")
    c = rep["chance"]
    print(f"\n[Repère du hasard] {c['n']} stratégies fournies sur les trois périodes : "
          f"{c['positive_a']} positives en A, {c['positive_b']} en B, {c['positive_c']} en C, "
          f"{c['positive_all']} sur les trois ; ROI moyen en C des 100 meilleures de A : "
          f"{_pct(c['roi_c_all_discovered'])}")
    print("\n[Les 10 meilleures de A]")
    for r in rep["top_discovery"]:
        _block(r)
    print(f"\n[{rep['n_validated']} des 100 meilleures de A restent positives en B ; "
          f"finalistes, IC en C à {100 * (1 - rep['alpha']):.1f} %]")
    for r in rep["finalists"]:
        _block(r)
    verdict = ("SUCCÈS : une stratégie reste significativement gagnante sur la période C"
               if rep["success"] else "ÉCHEC : aucune stratégie ne tient sur la période C")
    print(f"\nCritère fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
