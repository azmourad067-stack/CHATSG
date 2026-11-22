"""« Angles » de turfiste en simple gagnant : signaux connus que le marché pourrait mal payer.

Contrairement aux recherches précédentes (rang du modèle × cote × partants), chaque angle
est une hypothèse précise, fixée AVANT de voir les résultats (déferré des 4 pour la
première fois, œillères, forme du driver, descente de catégorie…), croisée avec une
tranche de cote et une discipline. Toutes les variables sont calculées à partir de courses
strictement antérieures (aucune fuite).

Protocole (identique à la grande recherche) : découverte sur la période A, validation sur
B, test final sur C jamais utilisé pour choisir, IC corrigé pour le nombre de finalistes.
En complément, un tableau descriptif de chaque angle seul sur toute la période, avec une
q-valeur de Benjamini-Hochberg (tient compte du nombre d'essais).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import grand_search as gs

Angle = Callable[[pd.DataFrame], pd.Series]


def _num(f: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(f[col], errors="coerce") if col in f else pd.Series(np.nan, f.index)


def _best_speed(f: pd.DataFrame) -> pd.Series:
    """Meilleure réduction kilométrique moyenne récente du lot (plus petite = plus vite)."""
    s = _num(f, "speed_red_mean3")
    rank = s.groupby(f["race_id"]).rank(method="min")
    return (rank == 1) & s.notna()


ANGLES: dict[str, Angle] = {
    "déferré des 4 pour la 1re fois": lambda f: _num(f, "shoes_d4_first") == 1,
    "déferré des 4": lambda f: _num(f, "shoes_d4") == 1,
    "œillères pour la 1re fois": lambda f: (_num(f, "blinkers_change") == 1)
    & (_num(f, "blinkers_on") == 1),
    "changement de driver / jockey": lambda f: _num(f, "rider_change") == 1,
    "driver / jockey en forme (≥ 20 % sur 30 j)": lambda f: (
        _num(f, "jockey_win_rate_30d") >= 0.20) & (_num(f, "jockey_runs_30d") >= 10),
    "entraîneur en forme (≥ 20 % sur 30 j)": lambda f: (
        _num(f, "trainer_win_rate_30d") >= 0.20) & (_num(f, "trainer_runs_30d") >= 10),
    "descente de catégorie": lambda f: _num(f, "class_change") <= -0.3,
    "montée de catégorie": lambda f: _num(f, "class_change") >= 0.3,
    "retour après ≥ 60 jours": lambda f: _num(f, "horse_days_since_db") >= 60,
    "recourt à ≤ 10 jours": lambda f: _num(f, "horse_days_since_db") <= 10,
    "a gagné sa dernière course": lambda f: _num(f, "mus_last_pos") == 1,
    "2e ou 3e de sa dernière course": lambda f: _num(f, "mus_last_pos").isin([2, 3]),
    # *_pct : rang d'arrivée moyen en percentile (0 = toujours vainqueur), lissé vers 0,5
    "réussit sur cet hippodrome": lambda f: (_num(f, "apt_course_runs") >= 3)
    & (_num(f, "apt_course_pct") <= 0.25),
    "réussit sur cette distance": lambda f: (_num(f, "apt_dist_runs") >= 3)
    & (_num(f, "apt_dist_pct") <= 0.25),
    "couple cheval-driver qui a déjà réussi": lambda f: (_num(f, "apt_pair_runs") >= 2)
    & (_num(f, "apt_pair_pct") <= 0.30),
    "meilleure vitesse récente du lot": _best_speed,
    "entraîneur performant sur cet hippodrome": lambda f: (
        _num(f, "trainer_course_runs") >= 5) & (_num(f, "trainer_course_win_rate") >= 0.20),
    "changement de distance ≥ 400 m": lambda f: _num(f, "dist_change").abs() >= 400,
    "favori des cotes": lambda f: _num(f, "mkt_rank") == 1,
    "modèle ≥ 1,3 × la cote": lambda f: _num(f, "_p_model") >= 1.3 * _num(f, "mkt_prob"),
}
ODDS_BANDS = ((1.0, 999.0), (1.0, 4.0), (4.0, 9.0), (9.0, 20.0), (20.0, 50.0))
DISCIPLINES = (None, "trot_attele", "plat")
DISC_LABELS = {None: "toutes disciplines", "trot_attele": "trot attelé", "plat": "plat"}
TOP_DISCOVERY = 20
N_FINAL = 5
MIN_BETS_A, MIN_BETS_B = 150, 50


def _odds_label(o: tuple[float, float]) -> str:
    return "toutes cotes" if o == (1.0, 999.0) else f"cote {o[0]:g} à {o[1]:g}"


def strategies() -> list[tuple[str, tuple[float, float], str | None]]:
    return list(product(ANGLES, ODDS_BANDS, DISCIPLINES))


def label(s: tuple[str, tuple[float, float], str | None]) -> str:
    return f"{s[0]}, {_odds_label(s[1])}, {DISC_LABELS[s[2]]}"


def masks(f: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: angle(f).fillna(False).to_numpy(bool) for name, angle in ANGLES.items()}


def strategy_mask(f: pd.DataFrame, angle_masks: dict[str, np.ndarray],
                  s: tuple[str, tuple[float, float], str | None]) -> np.ndarray:
    name, (lo, hi), disc = s
    odds = f["mkt_odds"].to_numpy(float)
    m = angle_masks[name] & (odds > lo) & (odds <= hi) & f["profit_gagnant"].notna().to_numpy()
    if disc is not None:
        m = m & (f["discipline"] == disc).to_numpy()
    return m


def _bets(f: pd.DataFrame, m: np.ndarray) -> pd.DataFrame:
    sub = f.loc[m, ["race_id", "race_date", "profit_gagnant"]]
    return sub.groupby(["race_id", "race_date"], as_index=False).agg(
        stake=("profit_gagnant", "size"), profit=("profit_gagnant", "sum"))


def _p_value(profit: np.ndarray) -> float:
    n = len(profit)
    sd = float(profit.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0:
        return 1.0
    z = float(profit.mean()) / (sd / math.sqrt(n))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def run(f: pd.DataFrame) -> dict[str, Any]:
    """`f` : un partant par ligne (prepare de profile_search + variables + `_p_model`)."""
    cuts = gs.cuts_of(f["race_date"])
    per = gs.periods(f["race_date"], cuts)
    profit = f["profit_gagnant"].to_numpy(float)
    am = masks(f)
    strats = strategies()
    counts, gains = [], []
    for s in strats:
        m = strategy_mask(f, am, s)
        counts.append(np.bincount(per[m], minlength=3))
        gains.append(np.bincount(per[m], weights=profit[m], minlength=3))
    c, g = np.array(counts, float), np.array(gains, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        roi = np.where(c > 0, g / c, np.nan)
    eligible = (c[:, 0] >= MIN_BETS_A) & (c[:, 1] >= MIN_BETS_B) & (c[:, 2] > 0)
    idx = np.flatnonzero(eligible)
    disc = idx[np.argsort(-roi[idx, 0], kind="stable")][:TOP_DISCOVERY]
    valid = disc[roi[disc, 1] > 0]
    final = valid[np.argsort(-roi[valid, 1], kind="stable")][:N_FINAL]
    alpha = 0.05 / max(1, len(final))

    def report(i: int, a: float = 0.05) -> dict[str, Any]:
        b = _bets(f, strategy_mask(f, am, strats[i]))
        p = gs.periods(b["race_date"], cuts) if not b.empty else np.array([], int)
        return {"label": label(strats[i]), "A": gs.evaluate(b[p == 0]),
                "B": gs.evaluate(b[p == 1]), "C": gs.evaluate(b[p == 2], alpha=a)}

    # tableau descriptif : chaque angle seul, toutes cotes, toutes disciplines, 2 ans
    rows = []
    for name in ANGLES:
        m = strategy_mask(f, am, (name, (1.0, 999.0), None))
        pr = profit[m]
        if len(pr) < 30:
            continue
        rows.append({"angle": name, "bets": len(pr), "roi": float(pr.mean()),
                     "hit_rate": float((pr > -1).mean()), "p": _p_value(pr),
                     "roi_A": float(np.nanmean(pr[per[m] == 0])) if (per[m] == 0).any()
                     else float("nan"),
                     "roi_C": float(np.nanmean(pr[per[m] == 2])) if (per[m] == 2).any()
                     else float("nan")})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["q"] = _bh(table["p"].to_numpy())
        table = table.sort_values("roi", ascending=False).reset_index(drop=True)
    big = eligible & (c[:, 2] >= MIN_BETS_B)
    pos = roi > 0
    finalists = [report(i, alpha) for i in final]
    return {
        "cuts": (str(cuts[0]), str(cuts[1])), "n_strategies": len(strats),
        "n_eligible": int(eligible.sum()), "alpha": alpha,
        "baseline": gs.evaluate(_bets(f, f["profit_gagnant"].notna().to_numpy())),
        "angles": table.to_dict("records"),
        "chance": {"n": int(big.sum()), "positive_a": int((pos[:, 0] & big).sum()),
                   "positive_b": int((pos[:, 1] & big).sum()),
                   "positive_c": int((pos[:, 2] & big).sum()),
                   "positive_all": int((pos.all(axis=1) & big).sum()),
                   "roi_c_discovered": float(np.nanmean(roi[disc, 2])) if len(disc)
                   else float("nan")},
        "top_discovery": [report(i) for i in disc[:5]],
        "n_validated": len(valid), "finalists": finalists,
        "success": any(r["C"].get("roi_ci", (0, 0))[0] > 0 for r in finalists),
    }
