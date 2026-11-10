"""Règles figées le 10/11/2026, AVANT d'avoir vu l'année de validation 2024-11 → 2025-11.

Elles ont été repérées sur les prédictions hors échantillon du 2025-11-20 au 2026-11-07
(stratégies par rang et analyse par segment). Elles ne doivent plus être modifiées :
leur seule utilité est d'être confrontées à une période qu'elles n'ont pas vue.

Critère de validation, fixé à l'avance : ROI > 0 sur la période de validation avec un
IC 95 % (bootstrap) entièrement au-dessus de zéro. Sinon la règle est écartée.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.betting_simulator import payout_lookup
from horse_intelligence.evaluation.segment_analysis import add_segments

FROZEN_ON = "2026-11-10"


@dataclass(frozen=True)
class FrozenRule:
    name: str
    rank: str            # rang du modèle, « 1 » à « 5 »
    bet_type: str        # SIMPLE_GAGNANT | SIMPLE_PLACE
    column: str | None   # colonne de segment (None : tout le sous-ensemble)
    value: str | None
    quinte_only: bool = False
    found_roi: float = 0.0   # ROI observé à la découverte (pour mémoire)


FROZEN_RULES = (
    FrozenRule("4e du modèle, placé, quinté", "4", "SIMPLE_PLACE", None, None, True, 0.122),
    FrozenRule("4e du modèle, gagnant, 1601-2100 m", "4", "SIMPLE_GAGNANT", "distance",
               "1601-2100 m", found_roi=0.072),
    FrozenRule("2e du modèle, placé, réunions R5+", "2", "SIMPLE_PLACE", "réunion", "R5+",
               found_roi=-0.017),
    FrozenRule("2e du modèle, gagnant, trot B", "2", "SIMPLE_GAGNANT", "groupe", "Trot B",
               found_roi=0.190),
    FrozenRule("3e du modèle, gagnant, trot G", "3", "SIMPLE_GAGNANT", "groupe", "Trot G",
               found_roi=0.063),
)


def evaluate_frozen_rules(frame: pd.DataFrame, payouts: pd.DataFrame, model_col: str,
                          n_boot: int = 4000, seed: int = 0) -> list[dict[str, Any]]:
    lookup = payout_lookup(payouts, "INTERNET")
    paid = {k[0] for k in lookup if k[1] == "SIMPLE_GAGNANT"}
    quinte = set(payouts.loc[payouts["bet_type"] == "QUINTE_PLUS", "race_id"].astype(int))
    f = add_segments(frame[frame["race_id"].isin(paid)], model_col)
    rng = np.random.default_rng(seed)
    out = []
    for rule in FROZEN_RULES:
        sub = f[f["rang_modèle"] == rule.rank]
        if rule.quinte_only:
            sub = sub[sub["race_id"].isin(quinte)]
        if rule.column is not None:
            sub = sub[sub[rule.column] == rule.value]
        profit = np.array([lookup.get((int(r), rule.bet_type, str(int(n))), 0.0) - 1.0
                           for r, n in zip(sub["race_id"], sub["num_pmu"], strict=True)])
        if len(profit) >= 2:
            boots = profit[rng.integers(0, len(profit), size=(n_boot, len(profit)))].mean(1)
            ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
        else:
            ci = (float("nan"), float("nan"))
        roi = float(profit.mean()) if len(profit) else float("nan")
        out.append({"rule": rule.name, "found_roi": rule.found_roi, "n_bets": len(profit),
                    "hit_rate": float((profit > -1).mean()) if len(profit) else float("nan"),
                    "roi": roi, "roi_ci": ci, "confirmed": bool(len(profit) and ci[0] > 0)})
    return out
