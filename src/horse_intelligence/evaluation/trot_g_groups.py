"""Règle trot G jouée sur plusieurs chevaux en simple gagnant (25/11/2026, idée de
l'utilisateur) : au lieu du seul 3e du modèle, jouer 2 ou 3 chevaux de la course.

Rappel : le résultat d'un groupe de chevaux joués 1 € chacun est la MOYENNE des résultats
de chaque rang. Grouper ne crée donc pas d'avantage par soi-même ; cela rend les gains plus
réguliers (plus souvent un gagnant), et cela n'améliore le résultat que si les rangs
ajoutés sont eux-mêmes rentables. Le tableau par rang (1er à 8e du modèle) l'explique.

Grille fixée AVANT de voir les résultats, sur les courses de trot G ouvertes au pari en
ligne (rapport Internet), avec les prédictions hors échantillon du modèle de production :
- groupes de rangs du modèle : {3} (règle actuelle, repère), {2,3}, {3,4}, {1,2,3},
  {2,3,4}, {3,4,5}, {2,3,4,5} ;
- deux façons de miser : 1 € par cheval, ou « dutching » (même mise totale, répartie en
  proportion de 1 / cote, pour toucher à peu près la même somme quel que soit le gagnant
  du groupe).
14 stratégies : succès si l'IC du ROI sur deux ans, corrigé pour 14 essais (Bonferroni),
est entièrement > 0. Résultat sur les six derniers mois donné à part.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import grand_search as gs
from horse_intelligence.evaluation.segment_analysis import race_group

GROUPS: tuple[tuple[int, ...], ...] = ((3,), (2, 3), (3, 4), (1, 2, 3), (2, 3, 4), (3, 4, 5),
                                       (2, 3, 4, 5))
STAKINGS = ("1 € par cheval", "dutching")
LAST_SIX_MONTHS = date(2026, 5, 23)


def trot_g(f: pd.DataFrame) -> pd.DataFrame:
    """Partants des courses de trot G réglées en simple gagnant (rapport Internet)."""
    groups = [race_group(lab, cond, disc) for lab, cond, disc in
              zip(f["label"], f["conditions_text"], f["discipline"], strict=True)]
    g = f[(np.array(groups) == "Trot G") & f["profit_gagnant"].notna().to_numpy()]
    return g.reset_index(drop=True)


def group_bets(f: pd.DataFrame, ranks: tuple[int, ...], staking: str) -> pd.DataFrame:
    """Une ligne par course : mise totale et gain net du groupe de rangs `ranks`."""
    sub = f[f["rank"].isin(ranks)].copy()
    # une course ne compte que si tous les rangs du groupe existent
    sub = sub[sub.groupby("race_id")["rank"].transform("size") == len(ranks)]
    if sub.empty:
        return pd.DataFrame(columns=["race_id", "race_date", "stake", "profit"])
    ret = sub["profit_gagnant"].to_numpy(float) + 1.0          # rapport pour 1 € (0 si perd)
    if staking == "dutching":
        inv = 1.0 / sub["mkt_odds"].to_numpy(float)
        share = inv / sub.assign(_i=inv).groupby("race_id")["_i"].transform("sum").to_numpy()
        stake = share * len(ranks)
    else:
        stake = np.ones(len(sub))
    sub = sub.assign(stake=stake, profit=stake * ret - stake)
    return sub.groupby(["race_id", "race_date"], as_index=False)[["stake", "profit"]].sum()


def by_rank(f: pd.DataFrame, max_rank: int = 8) -> list[dict[str, Any]]:
    rows = []
    for r in range(1, max_rank + 1):
        pr = f.loc[f["rank"] == r, "profit_gagnant"].to_numpy(float)
        if len(pr):
            rows.append({"rank": r, "bets": len(pr), "roi": float(pr.mean()),
                         "hit_rate": float((pr > -1).mean()),
                         "odds": float(f.loc[f["rank"] == r, "mkt_odds"].median())})
    return rows


def run(f: pd.DataFrame) -> dict[str, Any]:
    """`f` : un partant par ligne (profile_search.prepare + label, conditions_text)."""
    g = trot_g(f)
    alpha = 0.05 / (len(GROUPS) * len(STAKINGS))
    recent = pd.to_datetime(g["race_date"]).dt.date >= LAST_SIX_MONTHS
    out: list[dict[str, Any]] = []
    for ranks in GROUPS:
        for staking in STAKINGS:
            b = group_bets(g, ranks, staking)
            br = group_bets(g[recent.to_numpy()], ranks, staking)
            out.append({"label": f"rangs {'+'.join(map(str, ranks))}, {staking}",
                        "ranks": ranks, "staking": staking,
                        "all": gs.evaluate(b, alpha=alpha), "last6": gs.evaluate(br)})
    success = any(r["all"].get("roi_ci", (0, 0))[0] > 0 for r in out)
    return {"n_races": int(g["race_id"].nunique()), "alpha": alpha,
            "by_rank": by_rank(g), "strategies": out, "success": success}
