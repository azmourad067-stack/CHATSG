"""Prévision du rapport final (simple gagnant et simple placé) à l'instant de décision.

On parie au pari mutuel : le gain est le rapport FINAL, pas la cote de référence vue
10 minutes avant le départ. Les chevaux joués en dernière minute voient leur rapport
baisser. L'espérance d'un pari se calcule donc avec E[rapport | le cheval gagne (ou est
placé)], que l'on apprend sur les seuls chevaux gagnants (ou placés) des courses passées.

Protocole sans fuite : apprentissage par fenêtre croissante, mois par mois ; le modèle
qui prévoit un mois n'a vu que des courses des mois précédents. Les variables sont
toutes connues avant la décision (cote et rang de référence, partants,
discipline, probabilités hors échantillon des modèles).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.models.place_model import harville_place, placed_target

PAYOUT_PARAMS: dict[str, Any] = {
    "objective": "l2", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
    "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "verbose": -1, "seed": 7, "deterministic": True, "num_threads": 2,
}
PAYOUT_ROUNDS = 200
MIN_TRAIN_MONTHS = 3
MIN_BETS_SELECTION = 100


def payout_features(f: pd.DataFrame, win_col: str, place_col: str | None) -> pd.DataFrame:
    """Variables connues avant la décision."""
    odds = f["mkt_odds"].clip(lower=1.01)
    x = pd.DataFrame({
        "log_odds": np.log(odds),
        "mkt_rank": f["mkt_rank"],
        "n_runners": f["n_runners"],
        "model_vs_market": np.log(f[win_col].clip(lower=1e-4) / f["mkt_prob"].clip(lower=1e-4)),
        "trot": f["discipline"].astype(str).str.startswith("trot").astype(float),
        "plat": (f["discipline"].astype(str) == "plat").astype(float),
    }, index=f.index)
    if place_col is not None:
        x["harville_mkt_place"] = f["harville_mkt_place"]
        x["model_place_vs_market"] = np.log(f[place_col].clip(lower=1e-4)
                                            / f["harville_mkt_place"].clip(lower=1e-4))
    return x


def attach_payouts(pred: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    """Rapports officiels (masse Internet) du simple gagnant et du simple placé par cheval."""
    p = payouts[payouts["bet_type"].isin(["SIMPLE_GAGNANT", "SIMPLE_PLACE"])
                & payouts["payout_per_eur"].notna()].copy()
    p["num_pmu"] = pd.to_numeric(p["combination"], errors="coerce")
    p = p.dropna(subset=["num_pmu"]).astype({"num_pmu": int})
    wide = p.pivot_table(index=["race_id", "num_pmu"], columns="bet_type",
                         values="payout_per_eur", aggfunc="max").reset_index()
    wide = wide.rename(columns={"SIMPLE_GAGNANT": "payout_win", "SIMPLE_PLACE": "payout_place"})
    out = pred.merge(wide, on=["race_id", "num_pmu"], how="left")
    races_w = set(p.loc[p["bet_type"] == "SIMPLE_GAGNANT", "race_id"])
    races_p = set(p.loc[p["bet_type"] == "SIMPLE_PLACE", "race_id"])
    out["has_win_payouts"] = out["race_id"].isin(races_w)
    out["has_place_payouts"] = out["race_id"].isin(races_p)
    return out


def predict_payouts(f: pd.DataFrame, win_col: str, place_col: str | None,
                    min_train_months: int = MIN_TRAIN_MONTHS,
                    min_train_rows: int = 500) -> pd.DataFrame:
    """Ajoute `exp_payout_win` (et `exp_payout_place`) : rapport attendu SI le cheval gagne
    (est placé), prévu mois par mois avec un modèle appris sur les mois précédents."""
    import lightgbm as lgb

    f = f.copy()
    f["harville_mkt_place"] = harville_place(f, f["mkt_prob"])
    f["placed"] = placed_target(f)
    month = pd.to_datetime(f["race_date"]).dt.to_period("M")
    months = sorted(month.unique())
    targets: list[tuple[str, str, pd.Series, str | None]] = [
        ("exp_payout_win", "payout_win", f["won"] == 1, None)]
    if place_col is not None:
        targets.append(("exp_payout_place", "payout_place", f["placed"] == 1, place_col))
    for out_col, y_col, mask, pcol in targets:
        f[out_col] = np.nan
        x_all = payout_features(f, win_col, pcol)
        # rapport relatif à la cote de référence (gagnant) ou brut (placé), en log
        ref = np.log(f["mkt_odds"].clip(lower=1.01)) if y_col == "payout_win" else 0.0
        y_all = np.log(f[y_col].clip(lower=1.0)) - ref
        ok = mask & f[y_col].notna()
        for m in months[min_train_months:]:
            train = ok & (month < m)
            test = month == m
            if train.sum() < min_train_rows:
                continue
            booster = lgb.train(PAYOUT_PARAMS, lgb.Dataset(x_all[train], label=y_all[train]),
                                num_boost_round=PAYOUT_ROUNDS)
            resid = y_all[train] - booster.predict(x_all[train])
            smear = float(np.mean(np.exp(resid)))  # correction de retransformation (Duan)
            ref_t = ref[test].to_numpy() if isinstance(ref, pd.Series) else 0.0
            pred = np.asarray(booster.predict(x_all[test]), dtype=np.float64)
            f.loc[test, out_col] = np.exp(pred + ref_t) * smear
    return f


def payout_accuracy(f: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Erreur absolue en log du rapport gagnant : modèle de rapport contre cote de référence
    (appliquée telle quelle). Négatif = le modèle prévoit mieux le rapport final."""
    w = f[(f["won"] == 1) & f["payout_win"].notna() & f["exp_payout_win"].notna()]
    y = np.log(w["payout_win"])
    e_model = (y - np.log(w["exp_payout_win"])).abs().to_numpy()
    e_ref = (y - np.log(w["mkt_odds"])).abs().to_numpy()
    d = e_model - e_ref
    if d.size == 0:
        return {"n_winners": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    ratio = (w["payout_win"] / w["mkt_odds"]).to_numpy()
    return {"n_winners": int(d.size), "mae_model": float(e_model.mean()),
            "mae_reference_odds": float(e_ref.mean()), "diff": float(d.mean()),
            "ci": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
            "median_final_over_reference": float(np.median(ratio))}


@dataclass(frozen=True)
class EvRule:
    bet: str            # win | place
    ev_source: str      # predicted (rapport prévu) | reference (cote de référence)
    min_ev: float
    min_p: float
    max_odds: float = 1e9

    @property
    def name(self) -> str:
        src = "rapport prévu" if self.ev_source == "predicted" else "cote de référence"
        odds = "" if self.max_odds >= 1e9 else f", cote ≤ {self.max_odds:g}"
        return (f"{'gagnant' if self.bet == 'win' else 'placé'} · EV ({src}) ≥ "
                f"{self.min_ev:.2f}, P ≥ {self.min_p:.2f}{odds}")


def rule_bets(f: pd.DataFrame, rule: EvRule, win_col: str, place_col: str) -> pd.DataFrame:
    """Au plus un pari par course (meilleure EV) ; gain net au rapport officiel pour 1 €."""
    if rule.bet == "win":
        g = f[f["has_win_payouts"]].copy()
        g["p"] = g[win_col]
        pay = g["exp_payout_win"] if rule.ev_source == "predicted" else g["mkt_odds"]
        real = g["payout_win"]
    else:
        g = f[f["has_place_payouts"]].copy()
        g["p"] = g[place_col]
        pay = g["exp_payout_place"]
        real = g["payout_place"]
    g["ev"] = g["p"] * pay - 1.0
    g["profit"] = real.fillna(0.0) - 1.0
    g = g[g["ev"].notna() & (g["ev"] >= rule.min_ev) & (g["p"] >= rule.min_p)
          & (g["mkt_odds"] <= rule.max_odds)]
    g = g.sort_values(["race_id", "ev", "num_pmu"], ascending=[True, False, True])
    return g.drop_duplicates("race_id").sort_values(["race_date", "race_id"])


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


def rule_grid(bet: str, ev_source: str) -> list[EvRule]:
    if bet == "win":
        return [EvRule("win", ev_source, ev, mp, mo) for ev, mp, mo in
                product((0.0, 0.05, 0.10, 0.20), (0.0, 0.05, 0.10), (10.0, 30.0, 1e9))]
    return [EvRule("place", ev_source, ev, mp) for ev, mp in
            product((0.0, 0.03, 0.05, 0.10, 0.20), (0.1, 0.2, 0.3, 0.5))]


def select_then_test(f: pd.DataFrame, bet: str, ev_source: str, win_col: str,
                     place_col: str, selection_share: float = 0.5) -> dict[str, Any]:
    """Meilleure règle sur la 1re moitié (≥ 100 paris), appliquée telle quelle à la 2nde."""
    g = f[f[f"exp_payout_{bet}"].notna()]
    dates = sorted(g["race_date"].unique())
    if len(dates) < 2:
        return {"chosen": None, "selection": None, "test": None, "n_rules_tried": 0,
                "message": "Pas assez d'historique pour prévoir les rapports."}
    cut = dates[int(len(dates) * selection_share)]
    sel, test = g[g["race_date"] < cut], g[g["race_date"] >= cut]
    tried = [(r, summarize(rule_bets(sel, r, win_col, place_col)))
             for r in rule_grid(bet, ev_source)]
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
    out["test"] = summarize(rule_bets(test, best, win_col, place_col))
    out["message"] = ("Rentable sur le test, à confirmer (IC du ROI)." if out["test"]["roi"] > 0
                      else "La règle retenue en sélection n'est pas rentable sur le test.")
    return out
