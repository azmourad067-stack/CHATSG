"""Grande recherche de stratégies : simples et combinés, en trois périodes successives.

Familles :
- simple gagnant / placé : rang du modèle × cote de référence × discipline × partants ×
  filtre de valeur (aucun, modèle > marché, espérance ≥ +10 %, espérance ≥ +30 %) ;
- combinés (couplé gagnant, couplé placé, 2 sur 4, trio) : formules de 1 à 6 combinaisons
  sur le classement du modèle OU du marché (repère), × discipline × partants × ouverture
  de la course (probabilité du favori du classement).

L'espérance du simple est calculée sur la cote de référence corrigée du facteur 0,90
observé entre cote de référence et rapport final des gagnants.

Protocole fixé AVANT de voir les résultats, sur les dates triées :
1. période A (50 % des dates) : découverte. Les TOP_DISCOVERY meilleurs ROI (stratégies
   assez fournies) sont retenus ;
2. période B (25 % suivants) : validation. Parmi eux, ceux encore positifs ; les
   N_FINAL meilleurs ROI de B sont gardés ;
3. période C (25 % restants), jamais utilisée pour choisir : succès si l'IC du ROI d'une
   finaliste, corrigé pour N_FINAL essais (Bonferroni), est entièrement > 0.
L'IC est obtenu en rééchantillonnant les courses. Règlement au rapport définitif Internet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, cast

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.combo_simulator import winning_combos
from horse_intelligence.evaluation.profile_search import (
    DISCIPLINE_LABELS,
    DISCIPLINES,
    ODDS_RANGES,
    RANK_RANGES,
    RUNNER_RANGES,
)

PAYOUT_FACTOR = 0.90
VALUE_LEVELS = ("aucun", "modèle > marché", "espérance ≥ +10 %", "espérance ≥ +30 %")
OPENNESS = ((0.0, 1.0), (0.0, 0.25), (0.25, 0.40), (0.40, 1.0))
TOP_DISCOVERY = 100
N_FINAL = 5
MIN_BETS_A = {"simple": 200, "combo": 150}
MIN_BETS_B = 50

Ticket = tuple[int, ...]
# formule -> (pari, construction à partir du classement [n°1, n°2, …])
COMBO_FORMULAS: dict[str, tuple[str, Callable[[list[int]], list[Ticket]]]] = {
    "couplé gagnant 1-2": ("COUPLE_GAGNANT", lambda r: [(r[0], r[1])]),
    "couplé gagnant 1 avec 2, 3, 4": ("COUPLE_GAGNANT", lambda r: [(r[0], x) for x in r[1:4]]),
    "couplé gagnant 1-2-3 combiné": ("COUPLE_GAGNANT", lambda r: list(combinations(r[:3], 2))),
    "couplé placé 1-2": ("COUPLE_PLACE", lambda r: [(r[0], r[1])]),
    "couplé placé 1-2-3 combiné": ("COUPLE_PLACE", lambda r: list(combinations(r[:3], 2))),
    "couplé placé 1 avec 2 à 5": ("COUPLE_PLACE", lambda r: [(r[0], x) for x in r[1:5]]),
    "2 sur 4 1-2": ("DEUX_SUR_QUATRE", lambda r: [(r[0], r[1])]),
    "2 sur 4 1-2-3 combiné": ("DEUX_SUR_QUATRE", lambda r: list(combinations(r[:3], 2))),
    "trio 1-2-3": ("TRIO", lambda r: [(r[0], r[1], r[2])]),
    "trio 1-2-3-4 combiné": ("TRIO", lambda r: list(combinations(r[:4], 3))),
    "trio 1-2 avec 3 à 6": ("TRIO", lambda r: [(r[0], r[1], x) for x in r[2:6]]),
    "trio 1 avec 2 à 5 combiné": ("TRIO", lambda r: [(r[0], *c) for c in combinations(r[1:5], 2)]),
}


@dataclass(frozen=True)
class Strategy:
    family: str                 # "simple" | "combo"
    label: str
    key: tuple[Any, ...]


def _runner_label(rank: tuple[int, int], odds: tuple[float, float], disc: str | None,
                  runners: tuple[int, int]) -> str:
    r = "1er du modèle" if rank == (1, 1) else f"{rank[0]}e à {rank[1]}e du modèle"
    o = "toutes cotes" if odds == (1.0, 999.0) else f"cote {odds[0]:g} à {odds[1]:g}"
    return f"{r}, {o}, {DISCIPLINE_LABELS[disc]}, {_runners_label(runners)}"


def _runners_label(runners: tuple[int, int]) -> str:
    if runners == (0, 99):
        return "tous partants"
    if runners[0] == 0:
        return f"≤ {runners[1]} partants"
    if runners[1] == 99:
        return f"≥ {runners[0]} partants"
    return f"{runners[0]} à {runners[1]} partants"


def _open_label(o: tuple[float, float]) -> str:
    if o == (0.0, 1.0):
        return "toutes courses"
    if o[0] == 0.0:
        return f"favori < {100 * o[1]:.0f} %"
    if o[1] == 1.0:
        return f"favori ≥ {100 * o[0]:.0f} %"
    return f"favori {100 * o[0]:.0f}-{100 * o[1]:.0f} %"


Cuts = tuple[Any, Any]


def cuts_of(dates: pd.Series) -> Cuts:
    """Début de B (50 % des dates) et de C (75 %)."""
    uniq = sorted(dates.unique())
    return uniq[len(uniq) // 2], uniq[(3 * len(uniq)) // 4]


def periods(dates: pd.Series, cuts: Cuts) -> np.ndarray:
    """0 = A, 1 = B, 2 = C."""
    d = dates.to_numpy()
    return np.where(d < cuts[0], 0, np.where(d < cuts[1], 1, 2))


# --------------------------------------------------------------------- simples
def simple_scan(f: pd.DataFrame, model_col: str, cuts: Cuts
                ) -> tuple[list[Strategy], np.ndarray, np.ndarray]:
    """Pour chaque stratégie simple : nombre de paris et profit par période (3 colonnes)."""
    per = periods(f["race_date"], cuts)
    rank = f["rank"].to_numpy()
    odds = f["mkt_odds"].to_numpy(dtype=float)
    runners = f["n_runners"].to_numpy(dtype=float)
    disc = f["discipline"].to_numpy()
    p = f[model_col].to_numpy(dtype=float)
    ev = p * odds * PAYOUT_FACTOR - 1.0
    value = {"aucun": np.ones(len(f), bool), "modèle > marché": f["edge"].to_numpy(bool),
             "espérance ≥ +10 %": ev >= 0.10, "espérance ≥ +30 %": ev >= 0.30}
    rmask = {r: (rank >= r[0]) & (rank <= r[1]) for r in RANK_RANGES}
    omask = {o: (odds > o[0]) & (odds <= o[1]) for o in ODDS_RANGES}
    dmask = {d: np.ones(len(f), bool) if d is None else disc == d for d in DISCIPLINES}
    nmask = {n: (runners >= n[0]) & (runners <= n[1]) for n in RUNNER_RANGES}
    profits = {b: f[f"profit_{b}"].to_numpy(dtype=float) for b in ("gagnant", "placé")}
    settled = {b: ~np.isnan(v) for b, v in profits.items()}
    strategies, counts, gains = [], [], []
    for r, o, d, n, v, b in product(RANK_RANGES, ODDS_RANGES, DISCIPLINES, RUNNER_RANGES,
                                    VALUE_LEVELS, ("gagnant", "placé")):
        m = rmask[r] & omask[o] & dmask[d] & nmask[n] & value[v] & settled[b]
        counts.append(np.bincount(per[m], minlength=3))
        gains.append(np.bincount(per[m], weights=profits[b][m], minlength=3))
        filt = "" if v == "aucun" else f", {v}"
        strategies.append(Strategy("simple", f"{_runner_label(r, o, d, n)}{filt}, {b}",
                                   (r, o, d, n, v, b)))
    return strategies, np.array(counts, dtype=float), np.array(gains, dtype=float)


def simple_bets(f: pd.DataFrame, model_col: str, s: Strategy) -> pd.DataFrame:
    r, o, d, n, v, b = s.key
    odds = f["mkt_odds"].to_numpy(dtype=float)
    ev = f[model_col].to_numpy(dtype=float) * odds * PAYOUT_FACTOR - 1.0
    m = (f["rank"].between(*r) & (odds > o[0]) & (odds <= o[1])
         & f["n_runners"].between(*n) & f[f"profit_{b}"].notna()).to_numpy()
    if d is not None:
        m = m & (f["discipline"] == d).to_numpy()
    if v == "modèle > marché":
        m = m & f["edge"].to_numpy(bool)
    elif v != "aucun":
        m = m & (ev >= (0.10 if v.endswith("+10 %") else 0.30))
    sub = f.loc[m, ["race_id", "race_date", f"profit_{b}"]]
    return sub.groupby(["race_id", "race_date"], as_index=False).agg(
        stake=(f"profit_{b}", "size"), profit=(f"profit_{b}", "sum"))


# --------------------------------------------------------------------- combinés
def combo_races(pred: pd.DataFrame, payouts: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Une ligne par course : filtres de course et, pour chaque formule et classement
    (modèle / marché), mise et retour (NaN si le pari n'a pas été réglé)."""
    wins = winning_combos(payouts)
    rows = []
    for rid, g in pred.groupby("race_id", sort=False):
        rid_i = int(cast(int, rid))
        if len(g) < 6:
            continue
        nums = g["num_pmu"].to_numpy()
        row: dict[str, Any] = {"race_id": rid_i, "race_date": g["race_date"].iloc[0],
                               "discipline": g["discipline"].iloc[0],
                               "n_runners": float(g["n_runners"].iloc[0])}
        for src, col in (("modèle", model_col), ("marché", "mkt_prob")):
            p = g[col].to_numpy(dtype=float)
            order = [int(nums[i]) for i in np.argsort(-p, kind="stable")]
            row[f"fav_{src}"] = float(p.max() / p.sum())
            for name, (bet, build) in COMBO_FORMULAS.items():
                paid = wins.get((rid_i, bet))
                if paid is None:
                    row[f"{src}|{name}|stake"] = np.nan
                    row[f"{src}|{name}|ret"] = np.nan
                    continue
                tickets = [tuple(sorted(t)) for t in build(order)]
                row[f"{src}|{name}|stake"] = float(len(tickets))
                row[f"{src}|{name}|ret"] = float(sum(paid.get(t, 0.0) for t in tickets))
        rows.append(row)
    return pd.DataFrame(rows)


def combo_scan(races: pd.DataFrame, cuts: Cuts
               ) -> tuple[list[Strategy], np.ndarray, np.ndarray]:
    per = periods(races["race_date"], cuts)
    runners = races["n_runners"].to_numpy(dtype=float)
    disc = races["discipline"].to_numpy()
    dmask = {d: np.ones(len(races), bool) if d is None else disc == d for d in DISCIPLINES}
    nmask = {n: (runners >= n[0]) & (runners <= n[1]) for n in RUNNER_RANGES}
    strategies, counts, gains = [], [], []
    for src, name in product(("modèle", "marché"), COMBO_FORMULAS):
        stake = races[f"{src}|{name}|stake"].to_numpy(dtype=float)
        profit = races[f"{src}|{name}|ret"].to_numpy(dtype=float) - stake
        fav = races[f"fav_{src}"].to_numpy(dtype=float)
        ok = ~np.isnan(stake)
        for d, n, o in product(DISCIPLINES, RUNNER_RANGES, OPENNESS):
            m = ok & dmask[d] & nmask[n] & (fav >= o[0]) & (fav < o[1] + 1e-12)
            counts.append(np.bincount(per[m], weights=stake[m], minlength=3))
            gains.append(np.bincount(per[m], weights=profit[m], minlength=3))
            label = (f"{name} ({'classement du ' + src}), {DISCIPLINE_LABELS[d]}, "
                     f"{_runners_label(n)}, {_open_label(o)}")
            strategies.append(Strategy("combo", label, (src, name, d, n, o)))
    return strategies, np.array(counts, dtype=float), np.array(gains, dtype=float)


def combo_bets(races: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    src, name, d, n, o = s.key
    stake = races[f"{src}|{name}|stake"]
    m = stake.notna() & races["n_runners"].between(*n) & (races[f"fav_{src}"] >= o[0]) \
        & (races[f"fav_{src}"] < o[1] + 1e-12)
    if d is not None:
        m = m & (races["discipline"] == d)
    sub = races[m]
    return pd.DataFrame({"race_id": sub["race_id"], "race_date": sub["race_date"],
                         "stake": sub[f"{src}|{name}|stake"],
                         "profit": sub[f"{src}|{name}|ret"] - sub[f"{src}|{name}|stake"]})


# --------------------------------------------------------------------- évaluation
def evaluate(per_race: pd.DataFrame, alpha: float = 0.05, n_boot: int = 4000,
             seed: int = 0) -> dict[str, Any]:
    """ROI et IC (rééchantillonnage des courses) à partir d'une ligne par course."""
    if per_race.empty or per_race["stake"].sum() == 0:
        return {"bets": 0}
    pr = per_race.sort_values("race_date")
    stake, profit = pr["stake"].to_numpy(float), pr["profit"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pr), size=(n_boot, len(pr)))
    boots = profit[idx].sum(axis=1) / stake[idx].sum(axis=1)
    cum = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    days = max(1, (pd.to_datetime(pr["race_date"]).max()
                   - pd.to_datetime(pr["race_date"]).min()).days + 1)
    return {"bets": float(stake.sum()), "races": len(pr), "profit": float(profit.sum()),
            "roi": float(profit.sum() / stake.sum()),
            "roi_ci": (float(np.quantile(boots, alpha / 2)),
                       float(np.quantile(boots, 1 - alpha / 2))),
            "hit_rate": float((profit > -stake).mean()),
            "max_drawdown": float((peak - cum).max()),
            "stake_per_year": 365.0 * float(stake.sum()) / days}


def select(strategies: list[Strategy], counts: np.ndarray, gains: np.ndarray,
           top_discovery: int = TOP_DISCOVERY, n_final: int = N_FINAL) -> dict[str, Any]:
    """Étapes 1 et 2 du protocole ; renvoie aussi le repère du hasard."""
    with np.errstate(invalid="ignore", divide="ignore"):
        roi = np.where(counts > 0, gains / counts, np.nan)
    min_a = np.array([MIN_BETS_A[s.family] for s in strategies])
    eligible = (counts[:, 0] >= min_a) & (counts[:, 1] >= MIN_BETS_B) & (counts[:, 2] > 0)
    idx = np.flatnonzero(eligible)
    disc = idx[np.argsort(-roi[idx, 0], kind="stable")][:top_discovery]
    valid = disc[roi[disc, 1] > 0]
    final = valid[np.argsort(-roi[valid, 1], kind="stable")][:n_final]
    big = eligible & (counts[:, 2] >= MIN_BETS_B)
    pos = roi > 0
    return {
        "n_strategies": len(strategies), "n_eligible": int(eligible.sum()),
        "discovery": disc.tolist(), "validated": valid.tolist(), "final": final.tolist(),
        "roi": roi,
        "chance": {"n": int(big.sum()),
                   "positive_a": int((pos[:, 0] & big).sum()),
                   "positive_b": int((pos[:, 1] & big).sum()),
                   "positive_c": int((pos[:, 2] & big).sum()),
                   "positive_all": int((pos.all(axis=1) & big).sum()),
                   "roi_c_all_discovered": float(np.nanmean(roi[disc, 2])) if len(disc)
                   else float("nan")},
    }
