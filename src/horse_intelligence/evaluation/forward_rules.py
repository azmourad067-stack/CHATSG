"""Règles figées le 21/11/2026, jugées UNIQUEMENT sur les courses postérieures.

« Valeur 4e-6e » : 4e à 6e du modèle, cote de référence dans ]9 ; 20], 10 à 14 partants,
espérance du modèle ≥ +10 % (probabilité du modèle × cote × 0,90 − 1, le facteur 0,90
étant l'écart observé entre cote de référence et rapport final des gagnants), simple
gagnant, toutes disciplines, toutes réunions (nationales, régionales, locales) dès que les
paris en ligne sont ouverts, comme dans la grande recherche. Seule famille positive sur
les trois périodes de cette recherche (23/11/2024 → 15/11/2026) : +22,6 %, +41,8 %,
+22,0 % (IC du test final [−29 % ; +76 %]) ; NON significative après 8 640 essais. Elle
ne doit plus être modifiée.

Jugement : sur les prédictions enregistrées AVANT le départ (table `predictions`, la
dernière de chaque course), à partir du lendemain du gel ; règlement au rapport définitif
Internet (une course sans rapport Internet, donc sans pari en ligne, n'est pas comptée).
Critère fixé à l'avance : ROI > 0 avec un IC 95 % (rééchantillonnage des courses)
entièrement au-dessus de zéro, sur au moins MIN_BETS_VERDICT paris. Avant ce volume, le
bilan est seulement intermédiaire.

Correction du 22/11/2026, avant tout résultat de la période de suivi : l'application et le
bilan ne gardaient que les réunions nationales, alors que la règle a été trouvée sur toutes
les réunions ; ils suivent désormais la règle telle qu'elle a été testée.

Variante « tôt », figée le 23/11/2026 à la demande de l'utilisateur (VALUE_46_EARLY) :
même règle, mais décidée sur la dernière prédiction faite 45 à 60 min avant le départ
(et non la toute dernière). Sur les prédictions enregistrées du 01/11 au 23/11/2026,
comparaison faite APRÈS coup : 20 paris, +134 % (IC [−17 % ; +308 %]) contre 37 paris,
−40 % pour la décision 8 à 25 min avant. Non significatif ; suivie à part. C'est elle
que suivent les alertes SMS depuis le 23/11/2026.

« Petites courses, valeur » (SMALL_VALUE), figée le 24/11/2026 à la demande de
l'utilisateur : réunions régionales ou locales, ou allocation ≤ 10 000 € ; modèle de
production ; espérance ≥ +10 % (p × cote × 0,90 − 1), cote 2 à 20, simple gagnant. Règle
fixée AVANT le test walk-forward du 24/11/2026 (evaluation/small_meetings.py) : +15,4 %
sur deux ans (1 535 paris, IC 95 % [−2 % ; +35 %]), +8,0 % sur les six derniers mois ;
NON significative, et l'idée des petites courses vient des mêmes données. Le test décide
sur les cotes de référence publiées par le PMU 30 minutes avant le départ (vérifié sur
toutes les courses : 90 % publiées entre 30 et 35 min avant) : la décision est donc prise
sur la dernière prédiction faite 13 à 28 min avant le départ (après cette publication).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

MIN_BETS_VERDICT = 600


@dataclass(frozen=True)
class ValueRule:
    name: str
    frozen_on: date
    ranks: tuple[int, int]
    odds: tuple[float, float]       # cote de référence > odds[0] et ≤ odds[1]
    runners: tuple[int, int]
    min_ev: float
    payout_factor: float = 0.90
    bet_type: str = "SIMPLE_GAGNANT"
    # décision prise sur la dernière prédiction faite entre lead[0] (exclu) et lead[1]
    # (inclus) minutes avant le départ ; None : la toute dernière avant le départ
    lead_minutes: tuple[int, int] | None = None
    # petites courses seulement (réunion régionale ou locale, ou allocation ≤ SMALL_PRIZE)
    small_races: bool = False

    @property
    def first_day(self) -> date:
        return date.fromordinal(self.frozen_on.toordinal() + 1)


VALUE_46 = ValueRule("Valeur 4e-6e du modèle, cote 9 à 20, 10 à 14 partants, gagnant",
                     date(2026, 11, 21), (4, 6), (9.0, 20.0), (10, 14), 0.10)
VALUE_46_EARLY = ValueRule(
    "Valeur 4e-6e, décision 45 à 60 min avant le départ, gagnant", date(2026, 11, 23),
    (4, 6), (9.0, 20.0), (10, 14), 0.10, lead_minutes=(45, 60))
SMALL_PRIZE = 10000.0
SMALL_AUDIENCES = ("REGIONAL", "LOCAL")
SMALL_VALUE = ValueRule(
    "Petites courses, espérance ≥ +10 %, cote 2 à 20, gagnant", date(2026, 11, 24),
    (1, 99), (2.0, 20.0), (0, 99), 0.10, lead_minutes=(13, 28), small_races=True)
# même règle décidée 45 à 60 min avant le départ : suivie en arrière-plan seulement (ni
# SMS ni affichage), à la demande de l'utilisateur le 24/11/2026, pour comparer les deux
# moments de décision sur les courses à venir (novembre, avant gel : 32 paris, −11 %
# contre 61 paris, −35 % à 13-28 min ; écart non significatif)
SMALL_VALUE_EARLY = ValueRule(
    "Petites courses, décision 45 à 60 min avant (suivi seulement)", date(2026, 11, 24),
    (1, 99), (2.0, 20.0), (0, 99), 0.10, lead_minutes=(45, 60), small_races=True)


def is_small_race(audience: object, prize_eur: object) -> bool:
    """Réunion régionale ou locale, ou allocation ≤ SMALL_PRIZE € (petite course)."""
    prize = pd.to_numeric(pd.Series([prize_eur]), errors="coerce").iloc[0]
    return bool(str(audience) in SMALL_AUDIENCES
                or (pd.notna(prize) and float(prize) <= SMALL_PRIZE))


def rule_picks(table: pd.DataFrame, rule: ValueRule = VALUE_46) -> pd.DataFrame:
    """Chevaux de la règle dans UNE course : `num_pmu`, `p_win`, `mkt_odds` par partant.
    Aucun choix si une cote manque. Ajoute `model_rank` et `ev`."""
    n = len(table)
    if not rule.runners[0] <= n <= rule.runners[1] or table["mkt_odds"].isna().any():
        return table.iloc[0:0].assign(model_rank=pd.Series(dtype=int),
                                      ev=pd.Series(dtype=float))
    t = table.sort_values("p_win", ascending=False, kind="stable").reset_index(drop=True)
    t["model_rank"] = np.arange(1, n + 1)
    t["ev"] = t["p_win"] * t["mkt_odds"] * rule.payout_factor - 1.0
    lo, hi = rule.ranks
    return t[t["model_rank"].between(lo, hi) & (t["mkt_odds"] > rule.odds[0])
             & (t["mkt_odds"] <= rule.odds[1]) & (t["ev"] >= rule.min_ev)]


def _last_predictions(engine: Engine, first_day: date,
                      lead_minutes: tuple[int, int] | None = None) -> pd.DataFrame:
    """Partants de la DERNIÈRE prédiction enregistrée avant le départ de chaque course
    courue à partir de `first_day` (avec `lead_minutes` : la dernière faite entre
    lead[0] exclu et lead[1] inclus minutes avant le départ)."""
    query = text("""
        SELECT p.prediction_id, p.race_id, p.decision_time, r.race_date, r.audience,
               r.prize_eur,
               r.label, r.conditions_text, r.discipline, r.scheduled_start_utc
        FROM predictions p JOIN races r ON r.race_id = p.race_id
        WHERE r.race_date >= :first""")
    with engine.connect() as conn:
        preds = pd.read_sql(query, conn, params={"first": first_day})
        if preds.empty:
            return preds.assign(num_pmu=[], p_win=[], mkt_odds=[])
        entries = pd.read_sql(text(
            "SELECT e.prediction_id, e.num_pmu, e.p_win, e.mkt_odds "
            "FROM prediction_entries e JOIN predictions p "
            "ON p.prediction_id = e.prediction_id JOIN races r ON r.race_id = p.race_id "
            "WHERE r.race_date >= :first"), conn, params={"first": first_day})
    dt = pd.to_datetime(preds["decision_time"], utc=True)
    start = pd.to_datetime(preds["scheduled_start_utc"], utc=True)
    keep = dt < start
    if lead_minutes is not None:
        lead = (start - dt).dt.total_seconds() / 60
        keep = keep & (lead > lead_minutes[0]) & (lead <= lead_minutes[1])
    preds = preds[keep].assign(_dt=dt[keep])
    last = preds.sort_values("_dt").groupby("race_id").tail(1).drop(columns="_dt")
    last = last.assign(race_date=pd.to_datetime(last["race_date"]).dt.date)
    return last.merge(entries, on="prediction_id")


def forward_bets(engine: Engine, rule: ValueRule = VALUE_46,
                 national_only: bool = False) -> pd.DataFrame:
    """Paris de la règle sur les prédictions enregistrées avant le départ (dernière
    prédiction de chaque course), courses réglées seulement."""
    entries = _last_predictions(engine, rule.first_day, rule.lead_minutes)
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, combination, payout_per_eur FROM payouts "
            "WHERE pool = 'INTERNET' AND bet_type = :bt AND payout_per_eur IS NOT NULL"),
            conn, params={"bt": rule.bet_type})
    if national_only and not entries.empty:
        entries = entries[~entries["audience"].isin(("LOCAL", "REGIONAL"))]
    if rule.small_races and not entries.empty:
        entries = entries[[is_small_race(a, z) for a, z in
                           zip(entries["audience"], entries["prize_eur"], strict=True)]]
    settled = set(payouts["race_id"].astype(int))
    paid = {(int(r), str(c)): float(v) for r, c, v in zip(
        payouts["race_id"], payouts["combination"], payouts["payout_per_eur"], strict=True)}
    rows = []
    for rid, g in entries.groupby("race_id"):
        rid_i = int(cast(int, rid))
        if rid_i not in settled:
            continue
        for pick in rule_picks(g, rule).to_dict("records"):
            rows.append({"race_id": rid_i, "race_date": g["race_date"].iloc[0],
                         "num_pmu": int(pick["num_pmu"]), "mkt_odds": pick["mkt_odds"],
                         "profit": paid.get((rid_i, str(int(pick["num_pmu"]))), 0.0) - 1.0})
    return pd.DataFrame(rows, columns=["race_id", "race_date", "num_pmu", "mkt_odds",
                                       "profit"])


def races_with_internet_odds(engine: Engine, race_ids: list[int]) -> set[int]:
    """Courses dont des cotes Internet ont été publiées : le pari en ligne y est ouvert."""
    if not race_ids:
        return set()
    from sqlalchemy import select

    from horse_intelligence.data.storage import database as db

    o = db.odds_snapshots
    with engine.connect() as conn:
        rows = conn.execute(select(o.c.race_id).distinct().where(
            o.c.race_id.in_([int(r) for r in race_ids]), o.c.pool == "INTERNET")).all()
    return {int(r.race_id) for r in rows}


def forward_trot_g(engine: Engine, first_day: date = VALUE_46.first_day) -> pd.DataFrame:
    """Règle de la page (3e du modèle, gagnant, trot G) sur les mêmes prédictions
    d'avant-course, pour le même bilan. Toutes réunions depuis le 23/11/2026 (avant tout
    résultat de suivi) : une course sans rapport Internet, donc sans pari en ligne, n'est
    pas comptée."""
    from horse_intelligence.evaluation.segment_analysis import race_group

    entries = _last_predictions(engine, first_day)
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, combination, payout_per_eur FROM payouts WHERE pool = "
            "'INTERNET' AND bet_type = 'SIMPLE_GAGNANT' AND payout_per_eur IS NOT NULL"), conn)
    settled = set(payouts["race_id"].astype(int))
    paid = {(int(r), str(c)): float(v) for r, c, v in zip(
        payouts["race_id"], payouts["combination"], payouts["payout_per_eur"], strict=True)}
    rows = []
    for rid, g in entries.groupby("race_id"):
        rid_i = int(cast(int, rid))
        first = g.iloc[0]
        if (rid_i not in settled
                or race_group(first["label"], first["conditions_text"],
                              first["discipline"]) != "Trot G"
                or len(g) < 3 or g["mkt_odds"].isna().any()):
            continue
        pick = g.sort_values("p_win", ascending=False, kind="stable").iloc[2]
        rows.append({"race_id": rid_i, "race_date": first["race_date"],
                     "num_pmu": int(pick["num_pmu"]), "mkt_odds": pick["mkt_odds"],
                     "profit": paid.get((rid_i, str(int(pick["num_pmu"]))), 0.0) - 1.0})
    return pd.DataFrame(rows, columns=["race_id", "race_date", "num_pmu", "mkt_odds",
                                       "profit"])


@dataclass(frozen=True)
class D4StartRule:
    """« Déferré des 4 pour la 1re fois, autostart, place 4 à 6, cote ≤ 9 » (trot), figée le
    23/11/2026 à la demande de l'utilisateur. Seule finaliste de l'expérience d4_draw
    positive sur les trois périodes : +16,4 % (A, 222 paris), +26,2 % (B, 136), +17,7 % (C,
    123 ; IC corrigé [−27 % ; +66 %]) ; NON significative (60 combinaisons essayées).
    Place = numéro de corde publié, sinon numéro PMU (place derrière l'autostart)."""

    name: str
    frozen_on: date
    places: tuple[int, int]
    odds: tuple[float, float]       # cote de référence > odds[0] et ≤ odds[1]
    bet_type: str = "SIMPLE_GAGNANT"

    @property
    def first_day(self) -> date:
        return date.fromordinal(self.frozen_on.toordinal() + 1)


D4_AUTOSTART = D4StartRule("Déferré 4 (1re fois), autostart, place 4 à 6, cote ≤ 9, gagnant",
                           date(2026, 11, 23), (4, 6), (1.0, 9.0))


def is_autostart(conditions_text: object) -> bool:
    """Même définition que load_raw : « autostart » dans les conditions de la course."""
    return "autostart" in str(conditions_text or "").lower()


def start_place(draw: object, num_pmu: object) -> float:
    """Place au départ : numéro de corde publié, sinon numéro PMU."""
    d, n = pd.to_numeric(pd.Series([draw, num_pmu]), errors="coerce").tolist()
    return float(d) if pd.notna(d) else float(n)


def d4_rule_match(place: float, mkt_odds: float | None, autostart: bool, d4_first: bool,
                  rule: D4StartRule = D4_AUTOSTART) -> bool:
    if not (d4_first and autostart) or mkt_odds is None or pd.isna(mkt_odds):
        return False
    return (rule.places[0] <= place <= rule.places[1]
            and rule.odds[0] < float(mkt_odds) <= rule.odds[1])


def forward_d4(engine: Engine, rule: D4StartRule = D4_AUTOSTART) -> pd.DataFrame:
    """Paris de la règle déferré 4 / autostart sur les prédictions d'avant-course
    (dernière de chaque course), courses réglées seulement (rapport Internet)."""
    from horse_intelligence.prediction.signals import first_time_d4

    entries = _last_predictions(engine, rule.first_day)
    if entries.empty:
        return pd.DataFrame(columns=["race_id", "race_date", "num_pmu", "mkt_odds", "profit"])
    with engine.connect() as conn:
        payouts = pd.read_sql(text(
            "SELECT race_id, combination, payout_per_eur FROM payouts WHERE pool = "
            "'INTERNET' AND bet_type = 'SIMPLE_GAGNANT' AND payout_per_eur IS NOT NULL"), conn)
        draws = pd.read_sql(text(
            "SELECT race_id, num_pmu, draw, observed_at FROM race_entries WHERE race_id IN "
            "(SELECT race_id FROM races WHERE race_date >= :first)"), conn,
            params={"first": rule.first_day})
    draws = draws.sort_values("observed_at").drop_duplicates(["race_id", "num_pmu"],
                                                             keep="last")
    draw_of = {(int(r), int(n)): d for r, n, d in zip(draws["race_id"], draws["num_pmu"],
                                                      draws["draw"], strict=True)}
    d4: set[tuple[int, int]] = set()
    for day, g in entries.groupby("race_date"):
        d4 |= first_time_d4(engine, sorted({int(x) for x in g["race_id"]}),
                            cast(date, day))
    settled = set(payouts["race_id"].astype(int))
    paid = {(int(r), str(c)): float(v) for r, c, v in zip(
        payouts["race_id"], payouts["combination"], payouts["payout_per_eur"], strict=True)}
    rows = []
    for e in entries.to_dict("records"):
        rid, num = int(e["race_id"]), int(e["num_pmu"])
        if rid not in settled:
            continue
        place = start_place(draw_of.get((rid, num)), num)
        if d4_rule_match(place, e["mkt_odds"], is_autostart(e["conditions_text"]),
                         (rid, num) in d4, rule):
            rows.append({"race_id": rid, "race_date": e["race_date"], "num_pmu": num,
                         "mkt_odds": e["mkt_odds"],
                         "profit": paid.get((rid, str(num)), 0.0) - 1.0})
    return pd.DataFrame(rows, columns=["race_id", "race_date", "num_pmu", "mkt_odds",
                                       "profit"])


def verdict(bets: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> dict[str, Any]:
    if bets.empty:
        return {"bets": 0, "status": "aucun pari réglé"}
    per_race = bets.groupby("race_id")["profit"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_race), size=(n_boot, len(per_race)))
    pr, st = per_race["sum"].to_numpy(), per_race["size"].to_numpy()
    boots = pr[idx].sum(axis=1) / st[idx].sum(axis=1)
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    n = len(bets)
    if n < MIN_BETS_VERDICT:
        status = f"intermédiaire ({n} paris sur {MIN_BETS_VERDICT} nécessaires)"
    else:
        status = "VALIDÉE" if ci[0] > 0 else "NON validée"
    return {"bets": n, "profit": float(bets["profit"].sum()),
            "roi": float(bets["profit"].mean()), "roi_ci": ci,
            "hit_rate": float((bets["profit"] > -1).mean()),
            "period": (str(bets["race_date"].min()), str(bets["race_date"].max())),
            "status": status}
