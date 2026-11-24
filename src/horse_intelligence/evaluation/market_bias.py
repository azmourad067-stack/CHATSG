"""Le marché se trompe-t-il davantage dans certains contextes ? (24/11/2026)

Suite de la recherche par contexte (context_search), à la demande de l'utilisateur de
creuser la piste. La première recherche testait chaque petite case (contexte × rang ×
cote) SÉPARÉMENT : un biais réel mais petit, réparti sur beaucoup de cases, y reste
invisible, car chaque case a trop peu de paris. Ici, deux choses :

1. Carte du marché (descriptif, deux ans) : pour chaque contexte, perte en jouant tous
   les partants en gagnant et en placé, par tranche de cote, et apport du modèle sur le
   marché (gain de log-vraisemblance par partant : > 0 = le modèle prévoit mieux que les
   cotes dans ce contexte).

2. Stratégies « regroupées », fixées AVANT de voir les résultats (4 seulement, donc peu
   de pénalité statistique). Une régression ridge apprend, sur le passé, le profit moyen
   d'un pari selon TOUS les contextes du cheval à la fois (chaque contexte croisé avec la
   tranche de cote, et pour S3/S4 aussi avec le rang du modèle). Les contextes se
   recouvrent (une réunion régionale un mardi…) : la régression attribue chaque effet au
   bon contexte au lieu de le diluer. La pénalité ridge vaut SHRINK (l'équivalent de
   SHRINK paris fictifs à profit nul par case), ce qui écarte les cases chanceuses à petit
   effectif ; chaque pari est pondéré par l'inverse de la variance de son profit (~1/cote),
   pour que le bruit des gros outsiders ne domine pas. On joue ensuite, sur la période
   suivante jamais vue, chaque cheval dont le profit prévu est > 0 (une mise au plus).
   - S1 : contexte × tranche de cote, gagnant ;  S2 : idem, placé ;
   - S3 : contexte × tranche de cote × rang du modèle, gagnant ;  S4 : idem, placé.
   Critère principal : régression apprise sur A + B, jugée sur C ; succès si l'IC du ROI,
   corrigé pour 4 essais (Bonferroni), est entièrement > 0. En complément : apprise sur
   A seul, jugée sur B puis sur C.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from horse_intelligence.evaluation import context_search as cs
from horse_intelligence.evaluation import grand_search as gs

ODDS = ((1.0, 4.0), (4.0, 9.0), (9.0, 20.0), (20.0, 50.0))
RANKS = cs.RANKS
SHRINK = 300.0          # pénalité ridge : paris fictifs à profit nul par case
MIN_CELL_BETS = 30      # effectif minimal d'une case pour entrer dans la régression
ALL = "tous les partants"
STRATEGIES = {"S1": ("cote", "gagnant"), "S2": ("cote", "placé"),
              "S3": ("cote × rang", "gagnant"), "S4": ("cote × rang", "placé")}


def _band(values: np.ndarray, bands: tuple[tuple[float, float], ...]) -> np.ndarray:
    out = np.full(len(values), -1)
    for i, (lo, hi) in enumerate(bands):
        out[(values > lo) & (values <= hi)] = i
    return out


def _loglik(p: np.ndarray, won: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.where(won, np.log(p), np.log(1 - p))


def market_map(f: pd.DataFrame, ctx: dict[str, np.ndarray], model_col: str
               ) -> list[dict[str, Any]]:
    odds_band = _band(f["mkt_odds"].to_numpy(float), ODDS)
    won = (f["won"].fillna(0).to_numpy(float) > 0 if "won" in f
           else f["profit_gagnant"].to_numpy(float) > -1)
    gain = (_loglik(f[model_col].to_numpy(float), won)
            - _loglik(f["mkt_prob"].to_numpy(float), won))
    rows = []
    for name, m in ctx.items():
        row: dict[str, Any] = {"context": name, "runners": int(m.sum()),
                               "races": int(f.loc[m, "race_id"].nunique()),
                               "model_gain": float(gain[m].mean()) if m.any() else np.nan}
        for b in cs.BETS:
            pr = f[f"profit_{b}"].to_numpy(float)
            ok = m & ~np.isnan(pr)
            row[f"roi_{b}"] = float(pr[ok].mean()) if ok.any() else np.nan
            for i in range(len(ODDS)):
                okb = ok & (odds_band == i)
                row[f"roi_{b}_{i}"] = float(pr[okb].mean()) if okb.sum() >= 50 else np.nan
        rows.append(row)
    return rows


def _cells(f: pd.DataFrame, ctx: dict[str, np.ndarray], kind: str
           ) -> list[tuple[str, np.ndarray]]:
    odds_band = _band(f["mkt_odds"].to_numpy(float), ODDS)
    rank = f["rank"].to_numpy(float)
    rank_band = np.full(len(f), -1)
    for j, (lo, hi) in enumerate(RANKS):
        rank_band[(rank >= lo) & (rank <= hi)] = j
    out = []
    for name, m in {ALL: np.ones(len(f), bool), **ctx}.items():
        for i, (olo, ohi) in enumerate(ODDS):
            mo = m & (odds_band == i)
            if kind == "cote":
                out.append((f"{name}, cote {olo:g} à {ohi:g}", mo))
                continue
            for j, (rlo, rhi) in enumerate(RANKS):
                r = ("1er" if (rlo, rhi) == (1, 1) else f"{rlo}e et au-delà" if rhi == 99
                     else f"{rlo}e à {rhi}e")
                out.append((f"{name}, cote {olo:g} à {ohi:g}, {r} du modèle",
                            mo & (rank_band == j)))
    return out


def learn(cells: list[tuple[str, np.ndarray]], profit: np.ndarray, train: np.ndarray,
          weight: np.ndarray | None = None
          ) -> tuple[np.ndarray, list[tuple[str, int, float]]]:
    """Régression ridge du profit d'un pari sur les cases du cheval, apprise sur `train`.
    `weight` : poids de chaque pari (inverse de la variance du profit, ~1/cote : un
    outsider à 30 contre 1 apporte beaucoup de bruit pour peu d'information). Renvoie le
    profit prévu pour chaque partant et les cases (libellé, paris, effet) d'effet positif,
    de la plus forte à la plus faible."""
    ok = train & ~np.isnan(profit)
    if not ok.any():
        return np.full(len(profit), -np.inf), []
    w = np.ones(len(profit)) if weight is None else weight / weight[ok].mean()
    keep = [(lab, m) for lab, m in cells if int((m & ok).sum()) >= MIN_CELL_BETS]
    if not keep:
        return np.full(len(profit), -np.inf), []
    x = sparse.csc_matrix(np.column_stack([m for _, m in keep]).astype(np.float32))
    xt = x[ok]
    xw = sparse.diags(w[ok]) @ xt
    gram = (xt.T @ xw).toarray().astype(float)
    rhs = np.asarray(xw.T @ profit[ok]).ravel()
    coef = np.linalg.solve(gram + SHRINK * np.eye(len(keep)), rhs)
    pred = np.asarray(x @ coef).ravel()
    # un cheval hors de toute case apprise n'est pas joué
    pred[np.asarray(x.sum(axis=1)).ravel() == 0] = -np.inf
    n = np.asarray(xt.sum(axis=0)).ravel()
    effects = [(lab, int(k), float(c)) for (lab, _), k, c in zip(keep, n, coef, strict=True)
               if c > 0]
    return pred, sorted(effects, key=lambda e: -e[2])


def _bets(f: pd.DataFrame, mask: np.ndarray, col: str) -> pd.DataFrame:
    sub = f.loc[mask, ["race_id", "race_date", col]]
    return sub.groupby(["race_id", "race_date"], as_index=False).agg(
        stake=(col, "size"), profit=(col, "sum"))


def run(f: pd.DataFrame, model_col: str) -> dict[str, Any]:
    f = cs.with_context(f).reset_index(drop=True)
    cuts = gs.cuts_of(f["race_date"])
    per = gs.periods(f["race_date"], cuts)
    ctx = cs.contexts(f, per == 0)
    alpha = 0.05 / len(STRATEGIES)
    odds = f["mkt_odds"].to_numpy(float)
    strategies = {}
    for key, (kind, bet) in STRATEGIES.items():
        col = f"profit_{bet}"
        profit = f[col].to_numpy(float)
        # variance du profit d'un pari ~ cote (gagnant), ~ cote / 3 (placé) : poids inverse
        weight = 1.0 / np.maximum(odds if bet == "gagnant" else odds / 3.0, 1.0)
        cells = _cells(f, ctx, kind)
        res: dict[str, Any] = {"label": f"{key} : contexte × {kind}, {bet}",
                               "n_cells": len(cells)}
        for train_p, tests, tag in (((0, 1), (2,), "AB"), ((0,), (1, 2), "A")):
            train = np.isin(per, train_p)
            pred, effects = learn(cells, profit, train, weight)
            m = (pred > 0) & ~np.isnan(profit)
            out: dict[str, Any] = {"n_kept": len(effects),
                                   "top_cells": [{"label": lab, "bets": n, "roi_shrunk": c}
                                                 for lab, n, c in effects[:8]],
                                   "train": gs.evaluate(_bets(f, m & train, col))}
            for t in tests:
                a = alpha if (tag == "AB" and t == 2) else 0.05
                out["BC"[t - 1]] = gs.evaluate(_bets(f, m & (per == t), col), alpha=a)
            res[tag] = out
        strategies[key] = res
    success = any(s["AB"]["C"].get("roi_ci", (0, 0))[0] > 0 for s in strategies.values())
    return {"cuts": (str(cuts[0]), str(cuts[1])), "alpha": alpha,
            "map": market_map(f, ctx, model_col), "strategies": strategies,
            "success": success}
