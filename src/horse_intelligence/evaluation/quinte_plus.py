"""Quinté+ : combien de combinaisons jouer ? Combinés et champs réduits réglés aux rapports.

Une combinaison de 5 chevaux (mise de base 2 €) paie un seul rapport, le meilleur auquel
elle a droit :
- « ordre » : les 5 premiers dans l'ordre écrit ;
- « désordre » : les 5 premiers dans un autre ordre ;
- « bonus 4 » (ou « bonus 4sur5 ») : contient une des combinaisons de 4 payées ;
- « bonus 3 » : contient les 3 premiers.

Chaque combinaison est écrite dans l'ordre du classement utilisé (hypothèse prudente :
l'ordre n'est gagné que si cet ordre est exact). Les rapports sont pris tels que publiés
pour la masse Internet (par euro, × 2 € de mise de base).

Formules testées, pour un nombre de combinaisons croissant :
- combiné des N premiers : C(N, 5) combinaisons ;
- 1 base (le premier du classement) + N associés : C(N, 4) combinaisons.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import comb
from typing import Any, cast

import numpy as np
import pandas as pd

Ticket = tuple[int, ...]
BASE_STAKE = 2.0
RANKS = ("ordre", "desordre", "bonus4", "bonus3")


def quinte_results(payouts: pd.DataFrame, pool: str = "INTERNET") -> dict[int, dict[str, Any]]:
    """Course -> rapports payés par rang. Seules les courses avec ordre ET désordre sont
    gardées (rapports complets)."""
    p = payouts[(payouts["pool"] == pool) & payouts["payout_per_eur"].notna()
                & payouts["bet_type"].isin(["QUINTE_PLUS", "QUINTE_PLUS_DESORDRE"])]
    out: dict[int, dict[str, Any]] = {}
    for rid, bt, combo, val in zip(p["race_id"], p["bet_type"], p["combination"],
                                   p["payout_per_eur"], strict=True):
        parts = str(combo).split("-")
        if not all(x.isdigit() for x in parts):
            continue
        nums = tuple(int(x) for x in parts)
        d = out.setdefault(int(rid), {k: {} for k in RANKS})
        if bt == "QUINTE_PLUS_DESORDRE" and len(nums) == 5:
            d["desordre"][frozenset(nums)] = float(val)
        elif len(nums) == 5:
            d["ordre"][nums] = float(val)
        elif len(nums) == 4:
            d["bonus4"][frozenset(nums)] = float(val)
        elif len(nums) == 3:
            d["bonus3"][frozenset(nums)] = float(val)
    return {k: v for k, v in out.items() if v["ordre"] and v["desordre"]}


def ticket_rank(t: Ticket, paid: dict[str, Any]) -> tuple[str | None, float]:
    """Meilleur rang gagné par une combinaison et son rapport par euro."""
    if t in paid["ordre"]:
        return "ordre", paid["ordre"][t]
    s = frozenset(t)
    if s in paid["desordre"]:
        return "desordre", paid["desordre"][s]
    for rank in ("bonus4", "bonus3"):
        wins = [v for k, v in paid[rank].items() if k <= s]
        if wins:
            return rank, max(wins)
    return None, 0.0


def _ranked(nums: np.ndarray, score: np.ndarray) -> list[int]:
    return [int(nums[i]) for i in np.argsort(-score, kind="stable")]


def combined_tickets(nums: np.ndarray, score: np.ndarray, n: int) -> list[Ticket]:
    """Combiné des n premiers du classement : C(n, 5) combinaisons."""
    return list(combinations(_ranked(nums, score)[:n], 5))


def one_base_tickets(nums: np.ndarray, score: np.ndarray, n: int) -> list[Ticket]:
    """Le premier du classement en base + n associés : C(n, 4) combinaisons."""
    order = _ranked(nums, score)
    base, others = order[0], order[1:n + 1]
    pos = {h: i for i, h in enumerate(order)}
    return [tuple(sorted((base, *c), key=pos.__getitem__)) for c in combinations(others, 4)]


def n_tickets(formula: str, n: int) -> int:
    return comb(n, 5) if formula == "combine" else comb(n, 4)


def simulate(pred: pd.DataFrame, paid: dict[int, dict[str, Any]],
             make_tickets: Callable[[pd.DataFrame], list[Ticket]]) -> pd.DataFrame:
    """Une ligne par quinté : mise, retour et nombre de combinaisons gagnantes par rang."""
    rows = []
    for rid, g in pred.groupby("race_id", sort=False):
        rid_i = int(cast(int, rid))
        race = paid.get(rid_i)
        if race is None or len(g) < 5:
            continue
        tickets = make_tickets(g)
        counts = dict.fromkeys(RANKS, 0)
        ret = 0.0
        for t in tickets:
            rank, val = ticket_rank(t, race)
            if rank is not None:
                counts[rank] += 1
                ret += BASE_STAKE * val
        rows.append({"race_id": rid_i, "race_date": g["race_date"].iloc[0],
                     "stake": BASE_STAKE * len(tickets), "ret": ret, "hit": ret > 0,
                     "exact": counts["ordre"] > 0,
                     **{f"n_{k}": v for k, v in counts.items()}})
    cols = ["race_id", "race_date", "stake", "ret", "hit", "exact",
            *[f"n_{k}" for k in RANKS]]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["race_date", "race_id"]).reset_index(drop=True)


def rank_rates(res: pd.DataFrame) -> dict[str, float]:
    """Part des quintés où au moins une combinaison atteint chaque rang (meilleur rang)."""
    if res.empty:
        return {}
    best = np.select([res["n_ordre"] > 0, res["n_desordre"] > 0, res["n_bonus4"] > 0,
                      res["n_bonus3"] > 0], list(RANKS), default="rien")
    return {k: float((best == k).mean()) for k in (*RANKS, "rien")}
