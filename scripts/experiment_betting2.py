"""Expérience « betting2 » : paris combinés avec le modèle placé, et espérance calculée
avec le rapport final prévu au lieu de la cote de référence.

Usage : python scripts/experiment_betting2.py
Lecture seule ; utilise les prédictions hors échantillon du dernier entraînement.
Critère fixé AVANT de voir les résultats : une règle choisie sur la 1re moitié est rentable
sur la 2nde, IC 95 % du ROI entièrement > 0.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_betting2_experiment

LABELS = {"COUPLE_GAGNANT": "couplé gagnant", "COUPLE_PLACE": "couplé placé",
          "TRIO": "trio", "DEUX_SUR_QUATRE": "2 sur 4"}


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def _line(sel: dict | None, test: dict | None) -> str:
    s = "—" if not sel else f"sélection {_pct(sel['roi'])} ({sel['n_bets']} paris)"
    if test:
        s += (f" → test {_pct(test['roi'])} ({test['n_bets']} paris, IC "
              f"[{_pct(test['roi_ci'][0])} ; {_pct(test['roi_ci'][1])}])")
    return s


def main() -> int:
    configure_logging()
    rep = run_betting2_experiment(build_context().repo.engine)
    print(f"Période {rep['period'][0]} → {rep['period'][1]} : {rep['n_races']} courses, "
          f"{rep['seconds']:.0f} s (dont combinés {rep['seconds_combos']:.0f} s)")
    print("\n1. Paris combinés (règle choisie sur la 1re moitié, appliquée à la 2nde)")
    for col, by_bt in rep["combos"].items():
        src = "modèle placé" if col == "pl_place" else "modèle gagnant (avant)"
        print(f"  [{src}]")
        for bt, r in by_bt.items():
            refs = " ; ".join(f"{x['strategy']['name'].split(' ', 1)[1]} {_pct(x['roi'])}"
                              for x in r["references_test"])
            print(f"    {LABELS.get(bt, bt):15s} {_line(r['chosen_on_selection'], r['test'])}")
            print(f"    {'':15s} {r['message']} Repères (test) : {refs}")
    a = rep["payout_accuracy"]
    print("\n2. Rapport final prévu")
    if a.get("n_winners"):
        print(f"  Rapport final / cote de référence (gagnants, médiane) : "
              f"{a['median_final_over_reference']:.3f}")
        print(f"  Erreur moyenne (log) : modèle {a['mae_model']:.4f} vs cote de référence "
              f"{a['mae_reference_odds']:.4f} → écart {a['diff']:+.4f} "
              f"[{a['ci'][0]:+.4f} ; {a['ci'][1]:+.4f}] ({a['n_winners']} gagnants)")
    for name, r in rep["singles"].items():
        print(f"  [{name}] {r['chosen'] or '—'}")
        print(f"     {_line(r['selection'], r['test'])}")
        print(f"     {r['message']}")
    verdict = ("SUCCÈS : une règle est significativement rentable sur le test"
               if rep["success"] else "ÉCHEC : aucune règle significativement rentable")
    print(f"\nCritère fixé à l'avance (ROI du test, IC 95 % > 0) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
