"""Expérience « tuning » : réglage de LightGBM choisi sur la 1re moitié, jugé sur la 2nde.

Usage : python scripts/experiment_tuning.py
Critère fixé AVANT de voir les résultats : sur la 2nde moitié des plis, le réglage choisi
bat le réglage actuel en log-loss, IC 95 % entièrement négatif. Lecture seule.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import (
    TUNING_LARGE_CONFIGS,
    run_tuning_experiment,
)


def _ci(d: dict) -> str:
    return f"{d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]"


def main() -> int:
    configure_logging()
    large = "--large" in sys.argv
    rep = run_tuning_experiment(build_context().repo.engine,
                                configs=TUNING_LARGE_CONFIGS if large else None)
    print(f"{rep['n_races']} courses, {rep['n_folds']} plis, {rep['seconds']:.0f} s")
    for name, c in rep["configs"].items():
        print(f"  {name:13s} {c['num_rounds']} arbres {c['params'] or '(production)'}")
    print(f"\nSélection {rep['selection_period'][0]} → {rep['selection_period'][1]} "
          "(écart de log-loss vs réglage actuel, négatif = meilleur) :")
    for name, d in rep["first_half"].items():
        print(f"  {name:13s} {_ci(d)}")
    print(f"\nTest {rep['test_period'][0]} → {rep['test_period'][1]} :")
    for name, d in rep["second_half"].items():
        mark = "  ← choisi sur la sélection" if name == rep["chosen"] else ""
        print(f"  {name:13s} {_ci(d)}{mark}")
    for name, errs in rep["errors"].items():
        print(f"✖ {name}: {errs[:2]}")
    if large:
        ok = rep["success_both_halves"]
        verdict = ("SUCCÈS : « large » bat le réglage de production sur chaque moitié" if ok
                   else "ÉCHEC : « large » ne bat pas le réglage de production sur chaque moitié")
        print(f"\nCritère fixé à l'avance (« large » meilleur que la production sur CHAQUE "
              f"moitié, IC 95 % < 0) → {verdict}")
        return 0
    verdict = (f"SUCCÈS : le réglage « {rep['chosen']} » améliore le modèle" if rep["success"]
               else f"ÉCHEC : le réglage « {rep['chosen']} » ne fait pas significativement mieux")
    print(f"\nCritère fixé à l'avance (réglage choisi meilleur sur le test, IC 95 % < 0) "
          f"→ {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
