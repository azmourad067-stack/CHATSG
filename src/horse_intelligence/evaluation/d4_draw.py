"""« Déferré des 4 pour la 1re fois » × numéro de corde (place au départ), en simple gagnant.

Hypothèse de l'utilisateur (23/11/2026) : le signal « déferré des 4 pour la 1re fois »
(+0,2 % sur deux ans, contre −21,3 % pour l'ensemble des partants) deviendrait rentable
pour certaines places au départ. La place est le numéro de corde publié (`draw`) ; à
défaut (fréquent au trot), le numéro PMU, qui donne la place derrière l'autostart.

Grille fixée AVANT de voir les résultats : tranche de corde × type de départ (autostart,
volte, les deux) × tranche de cote = 60 combinaisons, uniquement sur les trotteurs
déferrés des 4 pour la 1re fois. Même protocole que la grande recherche : découverte A,
validation B, test final C jamais utilisé pour choisir, IC corrigé du nombre de finalistes.
Tableau descriptif en complément : chaque tranche de corde × départ, pour ces chevaux et,
en repère, pour les autres trotteurs.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import grand_search as gs
from horse_intelligence.evaluation.angles import _bets, _bh, _p_value

DRAW_BANDS = ((1, 99), (1, 3), (4, 6), (7, 9), (10, 99))
STARTS = (None, 1.0, 0.0)  # tous, autostart, volte
ODDS_BANDS = ((1.0, 999.0), (1.0, 9.0), (9.0, 20.0), (20.0, 50.0))
START_LABELS = {None: "tous départs", 1.0: "autostart", 0.0: "départ volté"}
TOP_DISCOVERY, N_FINAL = 10, 3
MIN_BETS_A, MIN_BETS_B = 100, 40

Strategy = tuple[tuple[int, int], float | None, tuple[float, float]]


def position(f: pd.DataFrame) -> pd.Series:
    """Place au départ : numéro de corde publié, sinon numéro PMU."""
    draw = pd.to_numeric(f["draw"], errors="coerce") if "draw" in f else pd.Series(
        np.nan, f.index)
    return draw.fillna(pd.to_numeric(f["num_pmu"], errors="coerce"))


def label(s: Strategy) -> str:
    (lo, hi), start, (olo, ohi) = s
    corde = "toutes cordes" if (lo, hi) == (1, 99) else (
        f"corde {lo} et plus" if hi == 99 else f"corde {lo} à {hi}")
    cote = "toutes cotes" if (olo, ohi) == (1.0, 999.0) else f"cote {olo:g} à {ohi:g}"
    return f"déferré 4 1re fois, {corde}, {START_LABELS[start]}, {cote}"


def _mask(f: pd.DataFrame, pos: np.ndarray, d4: np.ndarray, s: Strategy) -> np.ndarray:
    (lo, hi), start, (olo, ohi) = s
    odds = f["mkt_odds"].to_numpy(float)
    m = d4 & (pos >= lo) & (pos <= hi) & (odds > olo) & (odds <= ohi)
    m = m & f["profit_gagnant"].notna().to_numpy()
    if start is not None:
        m = m & (pd.to_numeric(f["autostart"], errors="coerce").to_numpy() == start)
    return m


def run(f: pd.DataFrame) -> dict[str, Any]:
    """`f` : un partant par ligne (profit_gagnant, mkt_odds, shoes_d4_first, draw,
    autostart, discipline, race_date)."""
    f = f[f["discipline"].astype(str).str.startswith("trot")].reset_index(drop=True)
    cuts = gs.cuts_of(f["race_date"])
    per = gs.periods(f["race_date"], cuts)
    profit = f["profit_gagnant"].to_numpy(float)
    pos = position(f).to_numpy(float)
    d4 = (pd.to_numeric(f["shoes_d4_first"], errors="coerce") == 1).to_numpy()
    strats: list[Strategy] = list(product(DRAW_BANDS, STARTS, ODDS_BANDS))
    counts, gains = [], []
    for s in strats:
        m = _mask(f, pos, d4, s)
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
        b = _bets(f, _mask(f, pos, d4, strats[i]))
        p = gs.periods(b["race_date"], cuts) if not b.empty else np.array([], int)
        return {"label": label(strats[i]), "A": gs.evaluate(b[p == 0]),
                "B": gs.evaluate(b[p == 1]), "C": gs.evaluate(b[p == 2], alpha=a)}

    rows = []
    for (band, start), group in product(product(DRAW_BANDS, STARTS), ("d4", "autres")):
        row_s: Strategy = (band, start, (1.0, 999.0))
        sel = d4 if group == "d4" else ~d4
        m = _mask(f, pos, sel, row_s)
        pr = profit[m]
        if len(pr) < 30:
            continue
        rows.append({"group": group, "label": label(row_s).replace(
            "déferré 4 1re fois, ", ""), "bets": len(pr), "roi": float(pr.mean()),
            "hit_rate": float((pr > -1).mean()), "p": _p_value(pr),
            "roi_A": float(pr[per[m] == 0].mean()) if (per[m] == 0).any() else np.nan,
            "roi_C": float(pr[per[m] == 2].mean()) if (per[m] == 2).any() else np.nan})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["q"] = _bh(table["p"].to_numpy())
    big = eligible & (c[:, 2] >= MIN_BETS_B)
    pos_roi = roi > 0
    finalists = [report(i, alpha) for i in final]
    return {
        "cuts": (str(cuts[0]), str(cuts[1])), "n_strategies": len(strats),
        "n_d4": int(d4.sum()),
        "draw_published_share": float(pd.to_numeric(f.loc[d4, "draw"], errors="coerce")
                                      .notna().mean()) if "draw" in f and d4.any()
        else 0.0,
        "table": table.to_dict("records"), "alpha": alpha,
        "chance": {"n": int(big.sum()), "positive_all": int((pos_roi.all(axis=1) & big).sum()),
                   "roi_c_discovered": float(np.nanmean(roi[disc, 2])) if len(disc)
                   else float("nan")},
        "top_discovery": [report(i) for i in disc[:5]], "n_validated": len(valid),
        "finalists": finalists,
        "success": any(r["C"].get("roi_ci", (0, 0))[0] > 0 for r in finalists),
    }
