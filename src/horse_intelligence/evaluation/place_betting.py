"""Simple placé : règles choisies sur une période de sélection, appliquées au test.

Il n'existe pas de cote placé publiée avant la course. Deux familles de règles :
- « sûr » : dans chaque course, le cheval de plus forte P(placé), joué si P ≥ seuil ;
- « valeur » : le cheval dont P(placé) du modèle dépasse le plus celle que donne le
  marché (Harville sur les cotes gagnant), joué si le rapport ≥ seuil et P ≥ minimum.
Règlement au rapport officiel de la masse Internet ; une course sans rapport placé
collecté est exclue. Seul le résultat du test compte.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

MIN_BETS_SELECTION = 100
SAFE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
VALUE_RATIOS = (1.05, 1.1, 1.2, 1.3, 1.5)
VALUE_MIN_P = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class PlaceRule:
    model: str
    kind: str          # safe | value
    min_p: float
    min_ratio: float = 0.0

    @property
    def name(self) -> str:
        if self.kind == "safe":
            return f"{self.model} : meilleur P(placé) ≥ {self.min_p:.2f}"
        return f"{self.model} : P/P_marché ≥ {self.min_ratio:.2f}, P ≥ {self.min_p:.2f}"


def place_payouts(payouts: pd.DataFrame) -> pd.DataFrame:
    p = payouts[(payouts["bet_type"] == "SIMPLE_PLACE") & payouts["payout_per_eur"].notna()]
    p = p.assign(num_pmu=pd.to_numeric(p["combination"], errors="coerce"))
    return p.dropna(subset=["num_pmu"]).astype({"num_pmu": int})[
        ["race_id", "num_pmu", "payout_per_eur"]].drop_duplicates(["race_id", "num_pmu"])


def rule_bets(pred: pd.DataFrame, pp: pd.DataFrame, rule: PlaceRule,
              market_col: str = "harville_market") -> pd.DataFrame:
    """Un pari au plus par course ; gain net pour 1 € au rapport officiel."""
    f = pred[pred["race_id"].isin(pp["race_id"].unique())].copy()
    f["p"] = f[rule.model]
    f["ratio"] = f["p"] / f[market_col].clip(lower=1e-6)
    f = f[f["p"] >= rule.min_p]
    if rule.kind == "value":
        f = f[f["ratio"] >= rule.min_ratio]
    score = "p" if rule.kind == "safe" else "ratio"
    f = f.sort_values(["race_id", score, "num_pmu"], ascending=[True, False, True])
    f = f.drop_duplicates("race_id")
    f = f.merge(pp, on=["race_id", "num_pmu"], how="left")
    f["profit"] = f["payout_per_eur"].fillna(0.0) - 1.0
    return f.sort_values(["race_date", "race_id"])


def summarize(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    p = bets["profit"].to_numpy()
    if len(p) == 0:
        return {"n_bets": 0, "roi": float("nan"), "roi_ci": (float("nan"), float("nan")),
                "hit_rate": float("nan")}
    rng = np.random.default_rng(seed)
    boots = p[rng.integers(0, len(p), size=(n_boot, len(p)))].mean(axis=1)
    return {"n_bets": len(p), "roi": float(p.mean()),
            "roi_ci": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
            "hit_rate": float((p > -1).mean())}


def rule_grid(model: str) -> list[PlaceRule]:
    rules = [PlaceRule(model, "safe", t) for t in SAFE_THRESHOLDS]
    if model != "harville_market":
        rules += [PlaceRule(model, "value", m, r) for r, m in product(VALUE_RATIOS, VALUE_MIN_P)]
    return rules


def select_then_test(pred: pd.DataFrame, payouts: pd.DataFrame, model: str,
                     selection_share: float = 0.5) -> dict[str, Any]:
    dates = sorted(pred["race_date"].unique())
    cut = dates[int(len(dates) * selection_share)]
    sel, test = pred[pred["race_date"] < cut], pred[pred["race_date"] >= cut]
    pp = place_payouts(payouts)
    tried = [(r, summarize(rule_bets(sel, pp, r))) for r in rule_grid(model)]
    eligible = [(r, s) for r, s in tried if s["n_bets"] >= MIN_BETS_SELECTION]
    out: dict[str, Any] = {"selection_period": (str(dates[0]), str(sel["race_date"].max())),
                           "test_period": (str(cut), str(dates[-1])),
                           "n_rules_tried": len(tried), "chosen": None, "selection": None,
                           "test": None}
    if not eligible:
        out["message"] = "Aucune règle n'a produit assez de paris en sélection."
        return out
    best, best_s = max(eligible, key=lambda t: t[1]["roi"])
    out.update(chosen=best.name, selection=best_s)
    if not best_s["roi"] > 0:
        out["message"] = ("Aucune règle n'est rentable sur la période de sélection : "
                          "aucune stratégie n'est retenue pour le test.")
        return out
    out["test"] = summarize(rule_bets(test, pp, best))
    out["message"] = ("Rentabilité observée sur le test, à confirmer (IC du ROI)."
                      if out["test"]["roi"] > 0 else
                      "La règle retenue en sélection n'est pas rentable sur le test.")
    return out
