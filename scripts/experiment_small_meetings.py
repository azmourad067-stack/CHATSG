"""Expérience « small_meetings » : modèle spécialisé pour les petites courses (réunions
régionales et locales, allocation ≤ 10 000 €) contre le modèle de production.

Usage : python scripts/experiment_small_meetings.py
Critères fixés AVANT de voir les résultats (evaluation/small_meetings.py). Lecture seule :
aucun modèle n'est enregistré.
"""

import sys
from typing import Any

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_small_meetings

LABELS = {"lgbm_market": "production", "ctx_cal": "combinaison spécifique",
          "small_only": "appris sur petites courses", "weighted": "petites courses × 3"}


def _pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:+.1f} %"


def _line(r: dict[str, Any]) -> str:
    if not r.get("bets"):
        return "aucun pari"
    return (f"{r['bets']:.0f} paris (~{r['stake_per_year']:.0f}/an), {r['profit']:+.0f} € "
            f"({_pct(r['roi'])}, IC [{_pct(r['roi_ci'][0])} ; {_pct(r['roi_ci'][1])}]), "
            f"gain sur {100 * r['hit_rate']:.0f} % des courses")


def _diff(d: dict[str, float]) -> str:
    return (f"{1000 * d['mean_diff']:+.2f} ‰ [{1000 * d['ci_low']:+.2f} ; "
            f"{1000 * d['ci_high']:+.2f}] ({d['n_races']} courses)")


def main() -> int:
    configure_logging()
    rep = run_small_meetings(build_context().repo.engine)
    print(f"Période {rep['period'][0]} → {rep['period'][1]} : {rep['n_races']} courses dont "
          f"{rep['n_small']} petites, {rep['n_folds']} plis, {rep['seconds']:.0f} s")
    if rep["errors"]:
        print(f"Erreurs : {rep['errors']}")
    print("\n[Log-loss contre le marché calibré, par course ; négatif = meilleur que les cotes]")
    for n, d in rep["small_vs_market"].items():
        print(f"  petites courses  {LABELS.get(n, n):28s} {_diff(d)}")
    for n, d in rep["big_vs_market"].items():
        print(f"  autres courses   {LABELS.get(n, n):28s} {_diff(d)}")
    print(f"\n[Variantes contre la production, petites courses ; IC à "
          f"{100 * (1 - rep['alpha_pred']):.1f} % ; négatif = la variante prévoit mieux]")
    for v, d in rep["variants_vs_prod"].items():
        print(f"  {LABELS[v]:28s} {_diff(d)}")
    print("  (autres courses, IC 95 %)")
    for v, d in rep["variants_vs_prod_big"].items():
        print(f"  {LABELS[v]:28s} {_diff(d)}")
    print("\n[Règle unique : espérance ≥ +10 %, cote 2 à 20, petites courses ; IC à "
          f"{100 * (1 - rep['alpha_bet']):.2f} % pour les variantes]")
    for b, r in rep["baseline"].items():
        print(f"  repère, tous les partants en {b} : {_line(r)}")
    for n, res in rep["betting"].items():
        for b in ("gagnant", "placé"):
            print(f"  {LABELS[n]:28s} {b:8s} toute la période : {_line(res[b]['all'])}")
            print(f"  {'':28s} {'':8s} 6 derniers mois  : {_line(res[b]['last6'])}")
    verdict = ("SUCCÈS : une variante est significativement gagnante sur les petites courses"
               if rep["success"] else "ÉCHEC : aucune variante n'est significativement "
               "gagnante sur les petites courses")
    print(f"\nCritère principal fixé à l'avance → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
