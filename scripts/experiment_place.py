"""Expérience « place » : un modèle appris directement sur « finit placé ».

Usage : python scripts/experiment_place.py
Compare, en walk-forward, la probabilité de place de production (Harville appliqué au
modèle gagnant) à un LightGBM appris sur la cible « placé ». Critère fixé AVANT de voir
les résultats : place_market ou place_stack bat harville_lgbm en log-loss « placé »,
IC 95 % entièrement négatif. Lecture seule : aucun modèle n'est enregistré.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import (
    PLACE_CHALLENGERS,
    PLACE_INCUMBENT,
    run_place_experiment,
)


def _ci(d: dict) -> str:
    return f"{d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]"


def main() -> int:
    configure_logging()
    rep = run_place_experiment(build_context().repo.engine)
    print(f"Période {rep['period'][0]} → {rep['period'][1]} : {rep['n_races']} courses "
          f"comparées, {rep['n_folds']} plis, {rep['seconds']:.0f} s")
    print(f"\n{'modèle':18s} {'logloss':>8s} {'brier':>7s} {'AUC':>6s} {'1er placé':>9s} "
          f"{'outsiders prévu/observé':>24s}")
    for name, m in rep["metrics"].items():
        print(f"{name:18s} {m['log_loss']:8.4f} {m['brier']:7.4f} {m['auc']:6.3f} "
              f"{m['top1_placed']:9.3f} {m['outsiders_pred']:11.3f} / {m['outsiders_obs']:.3f}")
    print("\nLog-loss « placé » vs Harville du marché (négatif = meilleur), IC 95 % :")
    for name, d in rep["vs_market"].items():
        print(f"  {name:18s} {_ci(d)}")
    print(f"\nLog-loss « placé » vs {PLACE_INCUMBENT} (méthode de production), IC 95 % :")
    for name, d in rep["vs_incumbent"].items():
        print(f"  {name:18s} {_ci(d)}")
    print("\nSimple placé (règle choisie sur la 1re moitié, appliquée à la 2nde) :")
    for name, b in rep["betting"].items():
        sel = b["selection"]
        line = f"  [{name}] {b['chosen'] or '—'}"
        if sel:
            line += f" : sélection {100 * sel['roi']:+.1f} % ({sel['n_bets']} paris)"
        t = b["test"]
        if t:
            line += (f" → test {100 * t['roi']:+.1f} % ({t['n_bets']} paris, IC "
                     f"[{100 * t['roi_ci'][0]:+.1f} % ; {100 * t['roi_ci'][1]:+.1f} %])")
        print(line)
        print(f"     {b['message']}")
    for name, errs in rep["errors"].items():
        print(f"✖ {name}: {errs[:2]}")
    verdict = ("SUCCÈS : le modèle placé améliore la probabilité de place" if rep["success"]
               else "ÉCHEC : le modèle placé ne fait pas significativement mieux")
    print(f"\nCritère fixé à l'avance ({' ou '.join(PLACE_CHALLENGERS)} meilleur que "
          f"{PLACE_INCUMBENT}, IC 95 % < 0) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
