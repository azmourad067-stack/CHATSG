"""Tiercé des courses du quinté : 20 combinaisons par course, réglées aux rapports officiels.

Le PMU ne propose pas de trio sur la course du quinté : l'équivalent est le tiercé. Une
combinaison de 3 chevaux (mise de base 1 €) paie le rapport « ordre » si les 3 premiers
arrivent dans l'ordre écrit, le rapport « désordre » s'ils arrivent dans un autre ordre.
Chaque combinaison est écrite dans l'ordre du modèle (le mieux classé d'abord).

Stratégies (20 combinaisons = 20 € par quinté) :
- `top6` : les 6 premiers d'un classement, en combiné (C(6,3) = 20 combinaisons) ;
- `proba20` : les 20 trios (sans ordre) les plus probables selon les forces de
  Plackett-Luce calées sur le modèle placé.

Décision sur probabilités hors échantillon ; règlement au rapport définitif de la masse
Internet. Une course sans rapport tiercé ordre ET désordre collecté est exclue.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any, cast

import numpy as np
import pandas as pd

Ticket = tuple[int, int, int]
N_TICKETS = 20


def tierce_results(payouts: pd.DataFrame, pool: str = "INTERNET"
                   ) -> dict[int, dict[str, Any]]:
    """Course -> arrivées payées : {"ordre": {(a, b, c): rapport}, "desordre":
    {frozenset: rapport}}. Seules les courses avec les deux rapports sont gardées."""
    p = payouts[(payouts["pool"] == pool) & payouts["payout_per_eur"].notna()
                & payouts["bet_type"].isin(["TIERCE", "TIERCE_DESORDRE"])]
    out: dict[int, dict[str, Any]] = {}
    for rid, bt, comb, val in zip(p["race_id"], p["bet_type"], p["combination"],
                                  p["payout_per_eur"], strict=True):
        parts = str(comb).split("-")
        if len(parts) != 3 or not all(x.isdigit() for x in parts):
            continue
        nums = tuple(int(x) for x in parts)
        d = out.setdefault(int(rid), {"ordre": {}, "desordre": {}})
        if bt == "TIERCE":
            d["ordre"][nums] = float(val)
        else:
            d["desordre"][frozenset(nums)] = float(val)
    return {k: v for k, v in out.items() if v["ordre"] and v["desordre"]}


def ticket_return(tickets: list[Ticket], paid: dict[str, Any]) -> tuple[float, bool]:
    """Somme rapportée par les combinaisons (1 € chacune) ; vrai si un ordre exact."""
    ret, exact = 0.0, False
    for t in tickets:
        if t in paid["ordre"]:
            ret += paid["ordre"][t]
            exact = True
        elif frozenset(t) in paid["desordre"]:
            ret += paid["desordre"][frozenset(t)]
    return ret, exact


def top6_tickets(nums: np.ndarray, score: np.ndarray) -> list[Ticket]:
    """Les 6 premiers du classement en combiné, chaque combinaison dans l'ordre du score."""
    order = [int(nums[i]) for i in np.argsort(-score, kind="stable")[:6]]
    return [(a, b, c) for a, b, c in combinations(order, 3)]


def proba20_tickets(nums: np.ndarray, strengths: np.ndarray, rank_score: np.ndarray,
                    rng: np.random.Generator, n_samples: int = 4000) -> list[Ticket]:
    """Les 20 trios les plus probables (Plackett-Luce sur `strengths`), écrits dans
    l'ordre de `rank_score`."""
    from horse_intelligence.evaluation.combo_simulator import combo_probabilities

    probs = combo_probabilities(nums, strengths, 3, 3, rng, n_samples)
    best = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:N_TICKETS]
    rank = {int(n): r for n, r in zip(nums, (-rank_score).argsort().argsort(), strict=True)}
    return [(a, b, c) for a, b, c in (tuple(sorted(c, key=lambda n: rank[n]))
                                      for c, _ in best)]


def simulate(pred: pd.DataFrame, paid: dict[int, dict[str, Any]],
             make_tickets: Callable[[pd.DataFrame], list[Ticket]]) -> pd.DataFrame:
    """Une ligne par quinté : mise, retour, trio trouvé, ordre exact."""
    rows = []
    for rid, g in pred.groupby("race_id", sort=False):
        rid_i = int(cast(int, rid))
        race = paid.get(rid_i)
        if race is None:
            continue
        tickets = make_tickets(g)
        ret, exact = ticket_return(tickets, race)
        rows.append({"race_id": rid_i,
                     "race_date": g["race_date"].iloc[0], "stake": float(len(tickets)),
                     "ret": ret, "hit": ret > 0, "exact": exact})
    cols = ["race_id", "race_date", "stake", "ret", "hit", "exact"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["race_date", "race_id"]).reset_index(drop=True)


def summarize(res: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> dict[str, Any]:
    if res.empty:
        return {"n_races": 0}
    stake, ret = res["stake"].to_numpy(), res["ret"].to_numpy()
    profit = ret - stake
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(res), size=(n_boot, len(res)))
    boots = profit[idx].sum(axis=1) / stake[idx].sum(axis=1)
    cum = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    streak = best = 0
    for h in res["hit"]:
        streak = 0 if h else streak + 1
        best = max(best, streak)
    hits = ret[ret > 0]
    return {"n_races": len(res), "staked": float(stake.sum()), "profit": float(profit.sum()),
            "roi": float(profit.sum() / stake.sum()),
            "roi_ci": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
            "hit_rate": float(res["hit"].mean()), "exact_rate": float(res["exact"].mean()),
            "median_hit_return": float(np.median(hits)) if len(hits) else float("nan"),
            "best_return": float(ret.max()), "longest_losing_streak": int(best),
            "max_drawdown": float((peak - cum).max())}


def by_period(res: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """1re moitié (dates), 2nde moitié (test), total."""
    if res.empty:
        return {}
    dates = sorted(res["race_date"].unique())
    cut = dates[len(dates) // 2]
    first, second = res[res["race_date"] < cut], res[res["race_date"] >= cut]
    first_end = str(first["race_date"].max()) if len(first) else ""
    return {"1re moitié": {**summarize(first), "period": (str(dates[0]), first_end)},
            "2nde moitié": {**summarize(second), "period": (str(cut), str(dates[-1]))},
            "total": summarize(res)}
