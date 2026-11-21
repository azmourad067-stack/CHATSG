"""Recherche de profils de course pour le simple gagnant / placé, sans se tromper soi-même.

Un profil combine : rang du modèle, cote de référence, discipline, nombre de partants,
avantage du modèle sur le marché (proba du modèle > proba du marché), type de pari.
Exemple : « 2e à 5e du modèle, cote 4 à 9, trot attelé, 10 à 14 partants, gagnant ».

Plusieurs milliers de profils sont essayés : certains sortent gagnants par pur hasard.
Protocole fixé AVANT de voir les résultats :
1. sélection sur la 1re moitié des dates seulement (profils d'au moins MIN_BETS_SELECTION
   paris), les TOP_K meilleurs ROI ;
2. jugement sur la 2nde moitié, jamais vue : succès si l'IC du ROI, corrigé pour TOP_K
   essais (Bonferroni), est entièrement > 0 ;
3. l'IC est calculé par rééchantillonnage des courses (plusieurs paris d'une même course
   ne sont pas indépendants).
Mise : 1 € par cheval retenu, réglé au rapport définitif de la masse Internet.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any, cast

import numpy as np
import pandas as pd

MIN_BETS_SELECTION = 200
TOP_K = 10

RANK_RANGES = ((1, 1), (1, 2), (1, 3), (2, 3), (2, 5), (3, 5), (4, 6))
ODDS_RANGES = ((1.0, 4.0), (4.0, 9.0), (9.0, 20.0), (20.0, 50.0), (4.0, 20.0), (1.0, 999.0))
DISCIPLINES = (None, "trot_attele", "trot_monte", "plat", "obstacle")
RUNNER_RANGES = ((0, 99), (0, 9), (10, 14), (15, 99))
BETS = ("gagnant", "placé")
BET_TYPES = {"gagnant": "SIMPLE_GAGNANT", "placé": "SIMPLE_PLACE"}
DISCIPLINE_LABELS = {None: "toutes disciplines", "trot_attele": "trot attelé",
                     "trot_monte": "trot monté", "plat": "plat", "obstacle": "obstacle"}


@dataclass(frozen=True)
class Profile:
    rank: tuple[int, int]
    odds: tuple[float, float]
    discipline: str | None
    runners: tuple[int, int]
    edge: bool
    bet: str

    def label(self) -> str:
        r = (f"{self.rank[0]}er du modèle" if self.rank == (1, 1)
             else f"{self.rank[0]}e à {self.rank[1]}e du modèle")
        o = ("toutes cotes" if self.odds == (1.0, 999.0)
             else f"cote {self.odds[0]:g} à {self.odds[1]:g}")
        n = ("tous partants" if self.runners == (0, 99)
             else f"≤ {self.runners[1]} partants" if self.runners[0] == 0
             else f"≥ {self.runners[0]} partants" if self.runners[1] == 99
             else f"{self.runners[0]} à {self.runners[1]} partants")
        e = ", modèle > marché" if self.edge else ""
        return f"{r}, {o}, {DISCIPLINE_LABELS[self.discipline]}, {n}{e}, {self.bet}"


EXAMPLE = Profile((2, 5), (4.0, 9.0), "trot_attele", (10, 14), False, "gagnant")
# profil affiché dans l'application (onglet « Outsiders attelé ») et ses variantes
OUTSIDERS = Profile((4, 6), (20.0, 50.0), "trot_attele", (15, 99), True, "gagnant")
FOLLOWED = (OUTSIDERS, replace(OUTSIDERS, bet="placé"),
            replace(OUTSIDERS, bet="placé", edge=False))


def profile_grid() -> list[Profile]:
    return [Profile(*p) for p in product(RANK_RANGES, ODDS_RANGES, DISCIPLINES,
                                         RUNNER_RANGES, (False, True), BETS)]


def prepare(frame: pd.DataFrame, payouts: pd.DataFrame, model_col: str,
            pool: str = "INTERNET") -> pd.DataFrame:
    """Une ligne par partant : rang du modèle, cote, profit de chaque pari (NaN si le
    pari n'a pas été réglé sur la course)."""
    p = payouts[(payouts["pool"] == pool) & payouts["payout_per_eur"].notna()]
    f = frame[frame["mkt_odds"].notna() & frame[model_col].notna()].copy()
    f["rank"] = f.groupby("race_id")[model_col].rank(ascending=False, method="first")
    f["edge"] = f[model_col] > f["mkt_prob"]
    f["discipline"] = f["discipline"].astype(object)
    keys = list(zip(f["race_id"].astype(int), f["num_pmu"].astype(int).astype(str),
                    strict=True))
    for bet, bt in BET_TYPES.items():
        q = p[p["bet_type"] == bt]
        settled = set(q["race_id"].astype(int))
        paid = {(int(r), str(c)): float(v)
                for r, c, v in zip(q["race_id"], q["combination"], q["payout_per_eur"],
                                   strict=True)}
        f[f"profit_{bet}"] = [paid.get(k, 0.0) - 1.0 if k[0] in settled else np.nan
                              for k in keys]
    return f.reset_index(drop=True)


def _mask(f: pd.DataFrame, cols: dict[str, np.ndarray], p: Profile) -> np.ndarray:
    m = ((cols["rank"] >= p.rank[0]) & (cols["rank"] <= p.rank[1])
         & (cols["odds"] > p.odds[0]) & (cols["odds"] <= p.odds[1])
         & (cols["runners"] >= p.runners[0]) & (cols["runners"] <= p.runners[1])
         & ~np.isnan(cols[f"profit_{p.bet}"]))
    if p.discipline is not None:
        m &= cols["discipline"] == p.discipline
    if p.edge:
        m &= cols["edge"]
    return m


def _columns(f: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"rank": f["rank"].to_numpy(), "odds": f["mkt_odds"].to_numpy(dtype=float),
            "runners": f["n_runners"].to_numpy(dtype=float),
            "discipline": f["discipline"].to_numpy(), "edge": f["edge"].to_numpy(dtype=bool),
            **{f"profit_{b}": f[f"profit_{b}"].to_numpy(dtype=float) for b in BETS}}


def scan(f: pd.DataFrame, profiles: list[Profile]) -> pd.DataFrame:
    """Nombre de paris et ROI de chaque profil sur `f`."""
    cols = _columns(f)
    rows = []
    for p in profiles:
        m = _mask(f, cols, p)
        n = int(m.sum())
        profit = cols[f"profit_{p.bet}"][m]
        rows.append({"profile": p, "bets": n,
                     "roi": float(profit.mean()) if n else np.nan,
                     "hit_rate": float((profit > -1).mean()) if n else np.nan})
    return pd.DataFrame(rows)


def evaluate(f: pd.DataFrame, p: Profile, alpha: float = 0.05, n_boot: int = 4000,
             seed: int = 0) -> dict[str, Any]:
    """ROI, IC (rééchantillonnage des courses, niveau 1 - alpha), réussite, rapport
    médian, pire série, creux maximal (en mises de 1 €)."""
    cols = _columns(f)
    m = _mask(f, cols, p)
    sub = f.loc[m, ["race_id", "race_date", f"profit_{p.bet}"]].rename(
        columns={f"profit_{p.bet}": "profit"})
    if sub.empty:
        return {"bets": 0}
    per_race = sub.groupby("race_id").agg(profit=("profit", "sum"), stake=("profit", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_race), size=(n_boot, len(per_race)))
    pr, st = per_race["profit"].to_numpy(), per_race["stake"].to_numpy()
    boots = pr[idx].sum(axis=1) / st[idx].sum(axis=1)
    sub = sub.sort_values(["race_date", "race_id"])
    profit = sub["profit"].to_numpy()
    cum = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    streak = worst = 0
    for x in profit:
        streak = 0 if x > -1 else streak + 1
        worst = max(worst, streak)
    wins = profit[profit > -1] + 1
    n_days = max(1, (pd.to_datetime(sub["race_date"]).max()
                     - pd.to_datetime(sub["race_date"]).min()).days + 1)
    return {"bets": len(sub), "races": len(per_race), "profit": float(profit.sum()),
            "roi": float(profit.mean()),
            "roi_ci": (float(np.quantile(boots, alpha / 2)),
                       float(np.quantile(boots, 1 - alpha / 2))),
            "hit_rate": float((profit > -1).mean()),
            "median_payout": float(np.median(wins)) if len(wins) else float("nan"),
            "longest_losing_streak": int(worst), "max_drawdown": float((peak - cum).max()),
            "bets_per_year": 365.0 * len(sub) / n_days}


def split_halves(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dates = sorted(f["race_date"].unique())
    cut = dates[len(dates) // 2]
    return f[f["race_date"] < cut], f[f["race_date"] >= cut], str(cut)


def search(f: pd.DataFrame, profiles: list[Profile] | None = None,
           top_k: int = TOP_K, min_bets: int = MIN_BETS_SELECTION) -> dict[str, Any]:
    """Sélection sur la 1re moitié, jugement sur la 2nde (protocole du module)."""
    profiles = profiles if profiles is not None else profile_grid()
    first, second, cut = split_halves(f)
    s1 = scan(first, profiles)
    s2 = scan(second, profiles)
    both = s1.merge(s2, on="profile", suffixes=("_1", "_2"))
    eligible = both[(both["bets_1"] >= min_bets) & (both["bets_2"] >= 1)]
    chosen = eligible.sort_values("roi_1", ascending=False).head(top_k)
    alpha = 0.05 / max(1, len(chosen))
    tested = [{"profile": p, "label": p.label(), "first": evaluate(first, p),
               "second": evaluate(second, p, alpha=alpha)} for p in chosen["profile"]]
    big = both[(both["bets_1"] >= min_bets) & (both["bets_2"] >= min_bets)]
    pos = big[(big["roi_1"] > 0) & (big["roi_2"] > 0)]
    return {
        "cut": cut, "period": (str(f["race_date"].min()), str(f["race_date"].max())),
        "n_profiles": len(profiles), "n_eligible": len(eligible), "alpha": alpha,
        "tested": tested,
        "success": any(t["second"].get("roi_ci", (0, 0))[0] > 0 for t in tested),
        # repère du hasard : part des profils (assez fournis) positifs sur les 2 moitiés
        "n_big": len(big),
        "n_positive_both": int(((big["roi_1"] > 0) & (big["roi_2"] > 0)).sum()),
        "n_positive_first": int((big["roi_1"] > 0).sum()),
        "n_positive_second": int((big["roi_2"] > 0).sum()),
        # pour information (sans valeur de preuve) : les profils positifs sur les 2 moitiés
        "positive_both": [
            {"label": prof.label(), "roi_1": float(r1), "roi_2": float(r2),
             "total": evaluate(f, prof)}
            for prof, r1, r2 in zip(cast(list[Profile], pos["profile"].tolist()),
                                    pos["roi_1"].tolist(), pos["roi_2"].tolist(), strict=True)],
        "followed": [{"label": p.label(), "first": evaluate(first, p),
                      "second": evaluate(second, p), "total": evaluate(f, p)}
                     for p in FOLLOWED],
        "example": {"label": EXAMPLE.label(), "first": evaluate(first, EXAMPLE),
                    "second": evaluate(second, EXAMPLE), "total": evaluate(f, EXAMPLE)},
    }
