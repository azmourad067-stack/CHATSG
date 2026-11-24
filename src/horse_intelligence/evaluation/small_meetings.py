"""Modèle spécialisé pour les « petites courses » (24/11/2026, à la demande de l'utilisateur).

Constat de départ (expérience market_bias) : le modèle prévoit nettement mieux que les cotes
dans les réunions régionales et locales et les courses à faible allocation (gain de
log-vraisemblance 3 à 4 fois plus fort qu'en réunion nationale). Le modèle de production
est pourtant appris et calibré sur toutes les courses ensemble : il donne au marché le
même poids partout, alors qu'en petite course le marché est moins fiable.

Petite course : réunion régionale ou locale, OU allocation ≤ SMALL_PRIZE €.

Trois variantes du modèle de production (mêmes variables, mêmes réglages), comparées à
lui en walk-forward (aucune fuite : chaque pli n'apprend que sur le passé) :
- `ctx_cal` : même LightGBM, mais la combinaison score / cotes est apprise séparément pour
  les petites courses (deux coefficients de plus dans le logit conditionnel) ;
- `small_only` : LightGBM appris sur les petites courses seulement, combinaison spécifique ;
- `weighted` : LightGBM où les petites courses pèsent SMALL_WEIGHT fois plus,
  combinaison spécifique.

Critères fixés AVANT de voir les résultats, sur les petites courses seulement :
1. prévision : une variante bat la production en log-loss (IC corrigé pour 3 variantes
   entièrement < 0) ;
2. argent (critère principal) : règle de mise unique, la même pour tous les modèles,
   « espérance ≥ +10 % (p × cote × 0,90 − 1), cote de 2 à 20 », en simple gagnant et en
   simple placé ; succès si l'IC du ROI d'une variante, corrigé pour 3 variantes × 2
   paris (Bonferroni, 6 essais), est entièrement > 0 sur toute la période hors
   échantillon. Le résultat sur les six derniers mois (après le 23/05/2026) est donné à
   part. L'idée venant de ces mêmes données (carte du marché), un succès devrait encore
   être confirmé sur les courses à venir avant toute mise réelle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import grand_search as gs
from horse_intelligence.models.lightgbm_model import LgbmRaceModel
from horse_intelligence.models.ranking import ConditionalLogit

SMALL_PRIZE = 10000.0
SMALL_WEIGHT = 3.0
SMALL_AUDIENCES = ("REGIONAL", "LOCAL")
VARIANTS = ("ctx_cal", "small_only", "weighted")
MIN_EV, ODDS = 0.10, (2.0, 20.0)
LAST_SIX_MONTHS = date(2026, 5, 23)


def is_small(frame: pd.DataFrame) -> np.ndarray:
    aud = frame["audience"].astype(str).isin(SMALL_AUDIENCES).to_numpy()
    prize = pd.to_numeric(frame["prize_eur"], errors="coerce").to_numpy(float)
    return aud | (prize <= SMALL_PRIZE)


@dataclass
class ContextLgbm(LgbmRaceModel):
    """LightGBM de production dont la combinaison score / cotes diffère pour les petites
    courses (colonne `small` du jeu de données, 0 ou 1)."""

    def _calib_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        z = super()._calib_inputs(frame)
        s = frame["small"].to_numpy(float)
        z["small_score"] = s * z["lgbm_score"]
        if self.with_market:
            z["small_mkt"] = s * z["mkt_log_prob"]
        return z

    def fit_weighted(self, base: pd.DataFrame, calib: pd.DataFrame,
                     weight: np.ndarray | None = None) -> ContextLgbm:
        import lightgbm as lgb

        self.categories = {c: sorted(map(str, base[c].dropna().astype("object").unique()))
                           for c in self.categorical}
        data = lgb.Dataset(self._matrix(base), label=base["won"].to_numpy(), weight=weight,
                           categorical_feature=self.categorical, free_raw_data=False)
        self.booster = lgb.train(self.params, data, num_boost_round=self.num_rounds)
        return self.calibrate(calib)

    def calibrate(self, calib: pd.DataFrame) -> ContextLgbm:
        z = self._calib_inputs(calib)
        self.calibrator = ConditionalLogit(l2=1e-3).fit(z, calib["race_id"], calib["won"],
                                                        list(z.columns))
        return self


def value_mask(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    ev = p * odds * gs.PAYOUT_FACTOR - 1.0
    return (ev >= MIN_EV) & (odds > ODDS[0]) & (odds <= ODDS[1])


def betting(f: pd.DataFrame, models: list[str], alpha: float) -> dict[str, Any]:
    """Règle de mise unique sur `f` (un partant par ligne, colonnes des modèles,
    mkt_odds, profit_gagnant, profit_placé, race_date)."""
    odds = f["mkt_odds"].to_numpy(float)
    recent = pd.to_datetime(f["race_date"]).dt.date.to_numpy() >= LAST_SIX_MONTHS
    out: dict[str, Any] = {}
    for name in models:
        m = value_mask(f[name].to_numpy(float), odds)
        res: dict[str, Any] = {}
        for bet in ("gagnant", "placé"):
            col = f"profit_{bet}"
            ok = m & f[col].notna().to_numpy()

            def per_race(mask: np.ndarray, col: str = col) -> pd.DataFrame:
                sub = f.loc[mask, ["race_id", "race_date", col]]
                return sub.groupby(["race_id", "race_date"], as_index=False).agg(
                    stake=(col, "size"), profit=(col, "sum"))

            res[bet] = {"all": gs.evaluate(per_race(ok), alpha=alpha),
                        "last6": gs.evaluate(per_race(ok & recent))}
        out[name] = res
    return out
