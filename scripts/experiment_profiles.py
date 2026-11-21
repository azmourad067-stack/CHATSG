"""Expérience « profiles » : recherche de profils de course pour le simple gagnant / placé.

Usage : python scripts/experiment_profiles.py
Protocole fixé AVANT de voir les résultats : 10 meilleurs profils choisis sur la 1re
moitié, jugés sur la 2nde avec un IC corrigé pour 10 essais. Lecture seule.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_profile_search


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def _line(r: dict[str, Any]) -> str:
    if not r.get("bets"):
        return "aucun pari"
    return (f"{r['bets']} paris (~{r['bets_per_year']:.0f}/an), résultat {r['profit']:+.0f} € "
            f"({_pct(r['roi'])}, IC [{_pct(r['roi_ci'][0])} ; {_pct(r['roi_ci'][1])}]), "
            f"réussite {100 * r['hit_rate']:.0f} %, rapport médian {r['median_payout']:.1f}, "
            f"pire série {r['longest_losing_streak']}, creux max {r['max_drawdown']:.0f} €")


def main() -> int:
    configure_logging()
    rep = run_profile_search(build_context().repo.engine)
    if not rep.get("n_races"):
        print("Aucune course réglée.")
        return 0
    print(f"{rep['n_races']} courses, {rep['period'][0]} → {rep['period'][1]} ; "
          f"1re moitié avant le {rep['cut']}, 2nde moitié à partir du {rep['cut']}")
    print(f"{rep['n_profiles']} profils essayés, {rep['n_eligible']} avec au moins 200 paris "
          f"sur la 1re moitié ; mise 1 € par cheval, rapport définitif Internet")
    ex = rep["example"]
    print(f"\n[Votre exemple : {ex['label']}]")
    for k, name in (("first", "1re moitié"), ("second", "2nde moitié"), ("total", "total")):
        print(f"  {name:12s} {_line(ex[k])}")
    for fol in rep["followed"]:
        print(f"\n[Suivi : {fol['label']}]")
        for k, name in (("first", "1re moitié"), ("second", "2nde moitié"), ("total", "total")):
            print(f"  {name:12s} {_line(fol[k])}")
    print(f"\n[Repère du hasard] parmi {rep['n_big']} profils fournis (≥ 200 paris sur chaque "
          f"moitié) : {rep['n_positive_first']} gagnants sur la 1re moitié, "
          f"{rep['n_positive_second']} sur la 2nde, {rep['n_positive_both']} sur les deux")
    for b in rep["positive_both"]:
        print(f"   • {b['label']} : 1re {_pct(b['roi_1'])}, 2nde {_pct(b['roi_2'])} ; "
              f"total {_line(b['total'])}")
    print(f"\n[Les {len(rep['tested'])} meilleurs profils de la 1re moitié, jugés sur la 2nde ; "
          f"IC à {100 * (1 - rep['alpha']):.1f} % (corrigé pour {len(rep['tested'])} essais)]")
    for i, t in enumerate(rep["tested"], 1):
        print(f"{i:2d}. {t['label']}")
        print(f"     1re moitié  {_line(t['first'])}")
        print(f"     2nde moitié {_line(t['second'])}")
    verdict = ("SUCCÈS : un profil reste significativement gagnant sur la 2nde moitié"
               if rep["success"] else "ÉCHEC : aucun profil ne tient sur la 2nde moitié")
    print(f"\nCritère fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
