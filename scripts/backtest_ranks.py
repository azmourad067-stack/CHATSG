"""Simple gagnant / placé sur le k-ième cheval (modèle ou marché), toutes courses ou quinté.

Usage : python scripts/backtest_ranks.py
Lecture seule ; prédictions hors échantillon du dernier entraînement, rapports officiels.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_rank_strategies


def _cell(r: dict) -> str:  # type: ignore[type-arg]
    if not r["n_bets"]:
        return f"{'—':>26s}"
    return f"{r['roi']:+7.1%} ({r['n_bets']:5d} p., {r['hit_rate']:4.0%})"


def main() -> int:
    configure_logging()
    rep = run_rank_strategies(build_context().repo.engine)
    print(f"Prédictions hors échantillon {rep['period'][0]} → {rep['period'][1]}, 1 € par course, "
          "rapport officiel. ROI (paris, réussite)")
    print(f"{'règle':38s} {'sélection (1re moitié)':>26s} {'TEST (2nde moitié)':>26s}"
          f"  {'test IC 95 %':>18s}")
    for r in rep["rules"]:
        lo, hi = r["test"]["roi_ci"]
        print(f"{r['rule']:38s} {_cell(r['selection'])} {_cell(r['test'])}  "
              f"[{lo:+.0%} ; {hi:+.0%}]")
    wins = rep["robust_winners"]
    print(f"\nRègles rentables sur la sélection ET le test (IC du test > 0) : {len(wins)}")
    for r in wins:
        print(f"  {r['rule']} : test {r['test']['roi']:+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
