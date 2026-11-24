"""Recherche par contexte de course : hippodrome, type de réunion, catégorie, distance,
allocation, jour de la semaine, conditions de sexe.

Idée (23/11/2026, conçue sans proposition de l'utilisateur) : les recherches précédentes
croisaient le rang du modèle, la cote, la discipline et le nombre de partants, mais jamais
le CONTEXTE de la course. Or les parieurs ne se trompent peut-être pas de la même façon
partout : petites masses des réunions régionales et locales, hippodromes où le public
connaît mal les chevaux venus d'ailleurs, catégories basses du trot, jours où les
parieurs occasionnels sont nombreux (week-end).

Grille fixée AVANT de voir les résultats : pour chaque contexte, 4 tranches de rang du
modèle × 4 tranches de cote de référence × 2 filtres de valeur (aucun, espérance ≥ +10 %)
× 2 paris (simple gagnant, simple placé) = 64 sélections. Les hippodromes retenus sont
ceux qui ont au moins MIN_HIPPO_RACES_A courses en période A (choisis sur A seulement).

Protocole identique à la grande recherche (grand_search.select) : découverte sur A
(100 meilleurs ROI parmi les stratégies assez fournies), validation sur B (encore
positives ; 5 meilleures gardées), test final sur C jamais utilisé pour choisir, IC
corrigé pour 5 essais (Bonferroni). Règlement au rapport définitif Internet.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import grand_search as gs
from horse_intelligence.evaluation.segment_analysis import race_group

RANKS = ((1, 1), (2, 3), (4, 6), (7, 99))
ODDS = ((1.0, 4.0), (4.0, 9.0), (9.0, 20.0), (20.0, 50.0))
VALUES = ("aucun", "espérance ≥ +10 %")
BETS = ("gagnant", "placé")
MIN_HIPPO_RACES_A = 150
AUDIENCES = {"NATIONAL": "nationale", "REGIONAL": "régionale", "LOCAL": "locale"}
DISTANCES = {"trot": ((0, 2100), (2101, 2700), (2701, 99999)),
             "plat": ((0, 1400), (1401, 2000), (2001, 99999)),
             "obstacle": ((0, 3500), (3501, 99999))}
PRIZES = ((0.0, 10000.0), (10000.0, 25000.0), (25000.0, 50000.0), (50000.0, 1e12))

Selection = tuple[tuple[int, int], tuple[float, float], str, str]


def _family(discipline: object) -> str:
    d = str(discipline or "")
    return "trot" if d.startswith("trot") else d


def _dist_label(fam: str, lo: int, hi: int) -> str:
    return (f"{fam}, {hi} m et moins" if lo == 0 else f"{fam}, plus de {lo - 1} m"
            if hi >= 99999 else f"{fam}, {lo} à {hi} m")


def _prize_label(lo: float, hi: float) -> str:
    k = lambda x: f"{x / 1000:g} k€"  # noqa: E731
    return (f"allocation ≤ {k(hi)}" if lo == 0 else f"allocation > {k(lo)}" if hi >= 1e12
            else f"allocation {k(lo)} à {k(hi)}")


def contexts(f: pd.DataFrame, in_a: np.ndarray) -> dict[str, np.ndarray]:
    """Masque (une valeur par partant) de chaque contexte. `in_a` : partants de la
    période A (seule utilisée pour choisir les hippodromes)."""
    out: dict[str, np.ndarray] = {}
    rc = f["racecourse"].fillna("?").astype(str).to_numpy()
    rid = f["race_id"].to_numpy()
    per_rc = pd.Series(rid[in_a]).groupby(rc[in_a]).nunique()
    for name in sorted(per_rc[per_rc >= MIN_HIPPO_RACES_A].index):
        out[f"hippodrome {name}"] = rc == name
    aud = f["audience"].astype(str).to_numpy()
    for code, label in AUDIENCES.items():
        out[f"réunion {label}"] = aud == code
    groups = f["group"].astype(str).to_numpy()
    for g in sorted(set(groups)):
        out[f"catégorie {g}"] = groups == g
    fam = f["discipline"].map(_family).to_numpy()
    dist = pd.to_numeric(f["distance_m"], errors="coerce").to_numpy(float)
    for name, bands in DISTANCES.items():
        for lo, hi in bands:
            out[_dist_label(name, lo, hi)] = (fam == name) & (dist >= lo) & (dist <= hi)
    prize = pd.to_numeric(f["prize_eur"], errors="coerce").to_numpy(float)
    for plo, phi in PRIZES:
        out[_prize_label(plo, phi)] = (prize > plo) & (prize <= phi)
    wd = pd.to_datetime(f["race_date"]).dt.day_of_week.to_numpy()
    out["samedi ou dimanche"] = wd >= 5
    out["lundi à vendredi"] = wd < 5
    sex = f["sex_condition"].fillna("").astype(str).to_numpy()
    for s in sorted(set(sex) - {""}):
        out[f"conditions de sexe {s}"] = sex == s
    return out


def selections() -> list[Selection]:
    return list(product(RANKS, ODDS, VALUES, BETS))


def _sel_label(s: Selection) -> str:
    (rlo, rhi), (olo, ohi), v, b = s
    r = ("1er du modèle" if (rlo, rhi) == (1, 1) else f"{rlo}e du modèle et au-delà"
         if rhi == 99 else f"{rlo}e à {rhi}e du modèle")
    val = "" if v == "aucun" else f", {v}"
    return f"{r}, cote {olo:g} à {ohi:g}{val}, {b}"


def _sel_mask(cols: dict[str, np.ndarray], s: Selection) -> np.ndarray:
    (rlo, rhi), (olo, ohi), v, b = s
    m = ((cols["rank"] >= rlo) & (cols["rank"] <= rhi) & (cols["odds"] > olo)
         & (cols["odds"] <= ohi) & ~np.isnan(cols[f"profit_{b}"]))
    if v != "aucun":
        m = m & (cols["ev"] >= 0.10)
    return m


def _cols(f: pd.DataFrame, model_col: str) -> dict[str, np.ndarray]:
    odds = f["mkt_odds"].to_numpy(float)
    return {"rank": f["rank"].to_numpy(float), "odds": odds,
            "ev": f[model_col].to_numpy(float) * odds * gs.PAYOUT_FACTOR - 1.0,
            **{f"profit_{b}": f[f"profit_{b}"].to_numpy(float) for b in BETS}}


def with_context(f: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la catégorie de course (`group`) à partir du libellé et des conditions."""
    return f.assign(group=[race_group(lab, cond, disc) for lab, cond, disc in
                           zip(f["label"], f["conditions_text"], f["discipline"],
                               strict=True)])


def run(f: pd.DataFrame, model_col: str) -> dict[str, Any]:
    """`f` : un partant par ligne (profile_search.prepare) + racecourse, audience, label,
    conditions_text, distance_m, prize_eur, sex_condition."""
    f = with_context(f).reset_index(drop=True)
    cuts = gs.cuts_of(f["race_date"])
    per = gs.periods(f["race_date"], cuts)
    ctx = contexts(f, per == 0)
    cols = _cols(f, model_col)
    sels = selections()
    sel_masks = [_sel_mask(cols, s) for s in sels]
    strategies, counts, gains = [], [], []
    for name, cm in ctx.items():
        for s, sm in zip(sels, sel_masks, strict=True):
            m = cm & sm
            profit = cols[f"profit_{s[3]}"]
            counts.append(np.bincount(per[m], minlength=3))
            gains.append(np.bincount(per[m], weights=profit[m], minlength=3))
            strategies.append(gs.Strategy("simple", f"{name} : {_sel_label(s)}", (name, s)))
    c, g = np.array(counts, float), np.array(gains, float)
    sel = gs.select(strategies, c, g)
    alpha = 0.05 / max(1, len(sel["final"]))

    def bets(i: int) -> pd.DataFrame:
        name, s = strategies[i].key
        m = ctx[name] & _sel_mask(cols, s)
        col = f"profit_{s[3]}"
        sub = f.loc[m, ["race_id", "race_date", col]]
        return sub.groupby(["race_id", "race_date"], as_index=False).agg(
            stake=(col, "size"), profit=(col, "sum"))

    def report(i: int, a: float = 0.05) -> dict[str, Any]:
        b = bets(i)
        p = gs.periods(b["race_date"], cuts) if not b.empty else np.array([], int)
        return {"label": strategies[i].label, "A": gs.evaluate(b[p == 0]),
                "B": gs.evaluate(b[p == 1]), "C": gs.evaluate(b[p == 2], alpha=a)}

    finalists = [report(i, alpha) for i in sel["final"]]
    kinds: dict[str, int] = {}
    for name in ctx:
        k = name.split(" ")[0]
        kinds[k] = kinds.get(k, 0) + 1
    return {
        "cuts": (str(cuts[0]), str(cuts[1])), "n_strategies": len(strategies),
        "n_contexts": len(ctx), "context_kinds": kinds, "n_eligible": sel["n_eligible"],
        "alpha": alpha, "chance": sel["chance"],
        "baseline": {b: gs.evaluate(f.loc[f[f"profit_{b}"].notna(),
                                          ["race_id", "race_date", f"profit_{b}"]]
                                    .groupby(["race_id", "race_date"], as_index=False)
                                    .agg(stake=(f"profit_{b}", "size"),
                                         profit=(f"profit_{b}", "sum")))
                     for b in BETS},
        "top_discovery": [report(i) for i in sel["discovery"][:10]],
        "n_validated": len(sel["validated"]), "finalists": finalists,
        "success": any(r["C"].get("roi_ci", (0, 0))[0] > 0 for r in finalists),
    }
