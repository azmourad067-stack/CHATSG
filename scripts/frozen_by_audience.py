"""Règle figée « 3e du modèle, gagnant, trot G » : réunions nationales contre régionales.

Usage : python scripts/frozen_by_audience.py
Lecture seule ; prédictions hors échantillon du dernier entraînement.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_frozen_by_audience


def main() -> int:
    configure_logging()
    rep = run_frozen_by_audience(build_context().repo.engine)
    print(f"Période {rep['period'][0]} → {rep['period'][1]} ; courses avec cotes : "
          f"{rep['n_races']}")
    for key, bet in (("rule", "gagnant"), ("rule_place", "placé")):
        print(f"\nRègle « 3e du modèle, {bet}, trot G » (lgbm_market) :")
        for name, periods in rep[key].items():
            parts = []
            for p, r in periods.items():
                if r["n_bets"]:
                    parts.append(
                        f"{p} {100 * r['roi']:+.1f} % ({r['n_bets']} paris, réussite "
                        f"{100 * r['hit_rate']:.0f} %, IC [{100 * r['roi_ci'][0]:+.1f} % ; "
                        f"{100 * r['roi_ci'][1]:+.1f} %])")
                else:
                    parts.append(f"{p} — (0 pari)")
            print(f"  {name:11s} " + " | ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
