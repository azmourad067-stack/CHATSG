"""Expérience « variables v2 » : vitesse, ferrure, aptitudes, forme récente.

Usage : python scripts/experiment_features.py
Critère fixé AVANT de voir les résultats : lgbm_market_v2 bat lgbm_market en log-loss,
IC 95 % entièrement négatif. Lecture seule : aucun modèle n'est enregistré.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_features_experiment


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def main() -> int:
    configure_logging()
    rep = run_features_experiment(build_context().repo.engine)
    print(f"Période {rep['period'][0]} → {rep['period'][1]} : {rep['n_races']} courses "
          f"comparées, {rep['n_folds']} plis, {rep['seconds']:.0f} s")
    print(f"\n{'modèle':16s} {'logloss':>8s} {'brier':>7s} {'top1':>6s} {'top3':>6s} "
          f"{'AUC':>6s} {'ECE':>6s}")
    for name, m in rep["metrics"].items():
        print(f"{name:16s} {m['log_loss']:8.4f} {m['brier']:7.4f} {m['top1']:6.3f} "
              f"{m['winner_in_top3']:6.3f} {m['auc']:6.3f} {m['ece']:6.4f}")
    print("\nDifférence de log-loss vs marché calibré (négatif = meilleur), IC 95 % :")
    for name, d in rep["vs_reference"].items():
        print(f"  {name:16s} {d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]")
    print("\nApport des variables v2 (v2 − v1, négatif = meilleur), IC 95 % :")
    for name, d in rep["v2_vs_v1"].items():
        print(f"  {name:16s} {d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]")
    for imp in rep["importance"].values():
        print("\nVariables les plus utilisées (score LightGBM v2, dernier pli) :")
        print("  " + ", ".join(f"{k} {v:.0f}" for k, v in imp.items()))
    print("\nRègles de mise (choisies sur la 1re moitié, appliquées à la 2nde) :")
    for name, b in rep["betting"].items():
        ch, t = b["chosen_on_selection"], b["test"]
        rule = b["chosen"]["name"] if b["chosen"] else "—"
        sel = f"{_pct(ch['roi'])} ({ch['n_bets']} paris)" if ch else "—"
        tst = (f"{_pct(t['roi'])} ({t['n_bets']} paris, IC [{_pct(t['roi_ci'][0])} ; "
               f"{_pct(t['roi_ci'][1])}])") if t else "—"
        print(f"  [{name}] {rule} : sélection {sel} → test {tst}")
        print(f"     {b['message']}")
    print(f"\nStratégies par rang (modèle v2) robustes : {len(rep['ranks_robust'])}")
    for r in rep["ranks_best_test"]:
        s, t = r["selection"], r["test"]
        print(f"  {r['rule']:34s} sélection {_pct(s['roi'])} ({s['n_bets']}) | test "
              f"{_pct(t['roi'])} ({t['n_bets']}) IC [{_pct(t['roi_ci'][0])} ; "
              f"{_pct(t['roi_ci'][1])}]")
    print("\nRègle figée « 3e du modèle, gagnant, trot G » selon le modèle :")
    for model, periods in rep["frozen_trot_g"].items():
        cells = []
        for name, r in periods.items():
            lo, hi = r["roi_ci"]
            cells.append(f"{name} {_pct(r['roi'])} ({r['n_bets']} paris, "
                         f"IC [{_pct(lo)} ; {_pct(hi)}])")
        print(f"  {model:15s} " + " | ".join(cells))
    for name, errs in rep["errors"].items():
        print(f"✖ {name}: {errs[:2]}")
    verdict = ("SUCCÈS : les variables v2 améliorent significativement le modèle"
               if rep["success"] else
               "ÉCHEC : les variables v2 n'améliorent pas significativement le modèle")
    print(f"\nCritère fixé à l'avance (lgbm_market_v2 meilleur que lgbm_market, IC 95 % < 0)"
          f" → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
