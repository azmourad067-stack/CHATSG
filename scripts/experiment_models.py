"""Expérience « models » : d'autres modèles que LightGBM binaire font-ils mieux ?

Usage : python scripts/experiment_models.py
Nécessite : pip install -r requirements-train.txt -r requirements-boost.txt
Candidats : LightGBM classement par course, CatBoost, empilement des trois scores.
Critère fixé AVANT de voir les résultats : un candidat bat lgbm_market en log-loss,
IC 95 % entièrement négatif. Lecture seule : aucun modèle n'est enregistré.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import (
    INCUMBENT,
    MODEL_CHALLENGERS,
    run_models_experiment,
)


def main() -> int:
    configure_logging()
    rep = run_models_experiment(build_context().repo.engine)
    print(f"Période {rep['period'][0]} → {rep['period'][1]} : {rep['n_races']} courses "
          f"comparées, {rep['n_folds']} plis, {rep['seconds']:.0f} s "
          f"({rep['n_fits']} apprentissages, CatBoost réappris {rep['catboost_refits']} fois)")
    print(f"\n{'modèle':16s} {'logloss':>8s} {'brier':>7s} {'top1':>6s} {'top3':>6s} "
          f"{'AUC':>6s} {'ECE':>6s}")
    for name, m in rep["metrics"].items():
        print(f"{name:16s} {m['log_loss']:8.4f} {m['brier']:7.4f} {m['top1']:6.3f} "
              f"{m['winner_in_top3']:6.3f} {m['auc']:6.3f} {m['ece']:6.4f}")
    print("\nDifférence de log-loss vs marché calibré (négatif = meilleur), IC 95 % :")
    for name, d in rep["vs_reference"].items():
        print(f"  {name:16s} {d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]")
    print(f"\nDifférence de log-loss vs {INCUMBENT} (modèle de production), IC 95 % :")
    for name, d in rep["vs_incumbent"].items():
        print(f"  {name:16s} {d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]")
    print("\nDurées d'apprentissage (premiers et derniers plis) :", rep["fits"])
    for name, errs in rep["errors"].items():
        print(f"✖ {name}: {errs[:2]}")
    verdict = ("SUCCÈS : un candidat bat le modèle de production" if rep["success"] else
               "ÉCHEC : aucun candidat ne bat significativement le modèle de production")
    print(f"\nCritère fixé à l'avance ({' ou '.join(MODEL_CHALLENGERS)} meilleur que "
          f"{INCUMBENT}, IC 95 % < 0) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
