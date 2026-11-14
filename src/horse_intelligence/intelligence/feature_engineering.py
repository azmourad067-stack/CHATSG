"""Construction des variables prédictives « as-of » (sans fuite de données).

Règles :
- l'instant de décision d'une course est `scheduled_start_utc - DECISION_OFFSET` ;
- l'historique interne (chevaux, jockeys, entraîneurs) n'utilise que les courses
  de dates STRICTEMENT antérieures ;
- les cotes de marché sont des cotes de référence publiées avant l'instant de décision ;
- les champs déclaratifs d'un engagement (cheval, monte, corde, poids, ferrure…)
  et les agrégats de la source (musique, cumuls de carrière) sont utilisés tels
  qu'observés : leur caractère « avant course » est vérifié par
  intelligence/leakage_audit.py à chaque entraînement ;
- les résultats ne servent qu'à la cible et à l'historique des courses ultérieures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from horse_intelligence.intelligence.musique import musique_features

DECISION_OFFSET = timedelta(minutes=10)
FEATURE_SET_VERSION = "fs-2026.11.2"

# Lissage bayésien empirique des taux de réussite (pseudo-observations)
SHRINK_RUNS = 20.0
# Taux de victoire vers lequel on lisse : constante fixée a priori (≈ 1 / partants moyens).
# Une moyenne calculée sur l'historique inclurait des résultats postérieurs à la course.
PRIOR_WIN_RATE = 0.08
CAREER_SHRINK = 5.0

CATEGORICAL = ["discipline", "racecourse", "sex", "blinkers", "shoes"]

FORM_FEATURES = [
    "age", "draw", "draw_rel", "weight_carried_kg", "weight_vs_field", "handicap_extra_m",
    "n_runners", "distance_m", "log_prize",
    "mus_n", "mus_last_pos", "mus_mean_pos_3", "mus_mean_pos_6", "mus_trend", "mus_wins_6",
    "mus_top3_6", "mus_incidents_6", "mus_same_disc_share",
    "career_starts", "career_win_rate", "career_place_rate", "log_earnings_per_start",
    "horse_db_runs", "horse_days_since_db", "horse_db_mean_pct",
    "jockey_win_rate", "jockey_db_runs", "trainer_win_rate", "trainer_db_runs",
    "pair_db_runs", "rider_change",
]
# Variables ramenées à la course (centrées dans la course) : utiles au logit conditionnel
RELATIVE_SOURCES = ["mus_mean_pos_6", "career_win_rate", "jockey_win_rate",
                    "trainer_win_rate", "log_earnings_per_start", "weight_carried_kg"]
RELATIVE_FEATURES = [f"{c}_rel" for c in RELATIVE_SOURCES]
MARKET_FEATURES = ["mkt_prob", "mkt_log_prob", "mkt_rank"]

# Variables « v2 » (11/2026), calculées à chaque construction mais utilisées par les modèles
# de production seulement si l'expérience `features` montre qu'elles améliorent la
# prévision (critère fixé avant de voir les résultats) :
# - vitesse (trot) : réduction kilométrique des courses passées, brute et comparée au
#   peloton de chaque course ;
# - ferrure / œillères : déferré des 4, changement depuis la dernière course ;
# - aptitudes : résultats passés du cheval sur l'hippodrome, la distance, le type de départ ;
# - forme récente des drivers / jockeys et entraîneurs sur 30 jours.
SPEED_FEATURES = ["speed_red_mean3", "speed_red_best5", "speed_vs_field_mean3", "speed_runs",
                  "speed_red_mean3_rel", "speed_vs_field_mean3_rel"]
GEAR_FEATURES = ["shoes_d4", "shoes_deferre", "shoes_change", "shoes_d4_first",
                 "blinkers_on", "blinkers_change"]
APTITUDE_FEATURES = ["apt_course_runs", "apt_course_pct", "apt_dist_runs", "apt_dist_pct",
                     "apt_start_runs", "apt_start_pct"]
RECENT_FEATURES = ["jockey_runs_30d", "jockey_win_rate_30d", "trainer_runs_30d",
                   "trainer_win_rate_30d"]
V2_FEATURES = SPEED_FEATURES + GEAR_FEATURES + APTITUDE_FEATURES + RECENT_FEATURES
# Variables « v3 » (11/2026), même règle d'adoption que les v2 (expérience `features3`) :
# - terrain (galop) : pénétromètre, piste en sable fibré (PSF), aptitude du cheval au terrain ;
# - changement de catégorie (dotation) et de distance par rapport aux courses précédentes ;
# - résultats passés du couple cheval / driver-jockey ; entraîneur sur l'hippodrome du jour.
V3_FEATURES = ["going_pen", "going_level", "going_psf", "apt_going_runs", "apt_going_pct",
               "class_change", "dist_change", "apt_pair_runs", "apt_pair_pct",
               "trainer_course_runs", "trainer_course_win_rate"]
D4 = "DEFERRE_ANTERIEURS_POSTERIEURS"
APT_SHRINK = 3.0
RECENT_DAYS = 30
RECENT_SHRINK = 5.0


@dataclass
class Dataset:
    frame: pd.DataFrame          # une ligne par partant (non-partants exclus)
    feature_set_version: str = FEATURE_SET_VERSION

    @property
    def model_features(self) -> list[str]:
        return FORM_FEATURES + RELATIVE_FEATURES

    def races(self) -> pd.Index:
        return pd.Index(self.frame["race_id"].unique())


def _read(engine: Engine, sql: str, **params: Any) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="mixed")


def load_raw(engine: Engine, until: date | None = None) -> dict[str, pd.DataFrame]:
    until = until or date(2100, 1, 1)
    races = _read(engine, """
        SELECT r.race_id, r.race_date, r.reunion_num, r.course_num, r.discipline,
               r.distance_m, r.prize_eur, r.declared_runners, r.scheduled_start_utc,
               r.result_final, r.country, rc.code AS racecourse,
               CASE WHEN LOWER(r.conditions_text) LIKE '%autostart%' THEN 1 ELSE 0 END
                   AS autostart
        FROM races r LEFT JOIN racecourses rc ON rc.racecourse_id = r.racecourse_id
        WHERE r.race_date <= :until""", until=until)
    entries = _read(engine, """
        SELECT e.* FROM race_entries e JOIN races r ON r.race_id = e.race_id
        WHERE r.race_date <= :until""", until=until)
    results = _read(engine, """
        SELECT res.race_id, res.num_pmu, res.finish_position, res.incident,
               res.reduction_km_s
        FROM race_results res JOIN races r ON r.race_id = res.race_id
        WHERE r.race_date <= :until""", until=until)
    going = _read(engine, """
        SELECT c.race_id, c.going_label, c.penetrometer, c.observed_at
        FROM racecourse_conditions c JOIN races r ON r.race_id = c.race_id
        WHERE r.race_date <= :until""", until=until)
    odds = _read(engine, """
        SELECT o.race_id, o.num_pmu, o.pool, o.odds, o.published_at, o.observed_at
        FROM odds_snapshots o JOIN races r ON r.race_id = o.race_id
        WHERE r.race_date <= :until AND o.odds_kind = 'reference'
          AND o.bet_type = 'SIMPLE_GAGNANT'""", until=until)
    for frame, cols in ((races, ["distance_m", "prize_eur", "declared_runners"]),
                        (entries, ["age", "draw", "weight_kg", "weight_carried_kg",
                                   "handicap_distance_m", "career_starts", "career_wins",
                                   "career_places", "career_earnings"]),
                        (results, ["finish_position", "reduction_km_s"]), (odds, ["odds"])):
        for c in cols:  # une colonne entièrement vide arrive en type objet
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")
    races["race_date"] = pd.to_datetime(races["race_date"]).dt.date
    races["scheduled_start_utc"] = _utc(races["scheduled_start_utc"])
    entries["observed_at"] = _utc(entries["observed_at"])
    odds["published_at"] = _utc(odds["published_at"])
    odds["observed_at"] = _utc(odds["observed_at"])
    going["observed_at"] = _utc(going["observed_at"])
    return {"races": races, "entries": entries, "results": results, "odds": odds,
            "going": going}


def _entries_at_decision(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Version de chaque engagement connue à l'instant de décision.

    Si aucune version n'a été observée avant la décision (collecte a posteriori),
    la plus ancienne observation est retenue : les champs déclaratifs ne dépendent pas
    de l'issue de la course (vérifié par l'audit de fuite).
    """
    e = entries.merge(races[["race_id", "decision_time"]], on="race_id")
    e = e.sort_values("observed_at")
    before = e[e["observed_at"] <= e["decision_time"]]
    latest_before = before.drop_duplicates(["race_id", "num_pmu"], keep="last")
    earliest = e.drop_duplicates(["race_id", "num_pmu"], keep="first")
    missing = earliest.merge(latest_before[["race_id", "num_pmu"]], how="left",
                             on=["race_id", "num_pmu"], indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    out = pd.concat([latest_before, missing], ignore_index=True)
    out["entry_seen_before_decision"] = out["observed_at"] <= out["decision_time"]
    return out


def _market(odds: pd.DataFrame, races: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    """Probabilités implicites des cotes de référence disponibles à la décision.

    Par course, une seule masse est utilisée (INTERNET de préférence, sinon OFFLINE) et
    seulement si tous les partants ont une cote ; sinon le marché est absent (NaN).
    """
    o = odds.merge(races[["race_id", "decision_time", "scheduled_start_utc"]], on="race_id")
    o = o[o["published_at"].notna() & (o["published_at"] < o["scheduled_start_utc"])
          & (o["published_at"] <= o["decision_time"])]
    o = o.sort_values("published_at").drop_duplicates(["race_id", "num_pmu", "pool"],
                                                      keep="last")
    o = o.merge(runners[["race_id", "num_pmu"]], on=["race_id", "num_pmu"])
    n_runners = runners.groupby("race_id").size().rename("n")
    cover = o.groupby(["race_id", "pool"]).size().rename("k").reset_index()
    cover = cover.merge(n_runners, left_on="race_id", right_index=True)
    cover = cover[cover["k"] == cover["n"]]
    cover["pref"] = (cover["pool"] != "INTERNET").astype(int)
    chosen = cover.sort_values("pref").drop_duplicates("race_id")[["race_id", "pool"]]
    o = o.merge(chosen, on=["race_id", "pool"])
    o["implied"] = 1.0 / o["odds"]
    total = o.groupby("race_id")["implied"].transform("sum")
    o["mkt_prob"] = o["implied"] / total
    o["mkt_overround"] = total
    o["mkt_log_prob"] = np.log(o["mkt_prob"])
    o["mkt_rank"] = o.groupby("race_id")["odds"].rank(method="min")
    return o[["race_id", "num_pmu", "pool", "odds", "mkt_prob", "mkt_log_prob", "mkt_rank",
              "mkt_overround"]].rename(columns={"pool": "mkt_pool", "odds": "mkt_odds"})


def _asof_before(df: pd.DataFrame, hist: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Joint à chaque ligne l'état d'historique de la DERNIÈRE date strictement antérieure.

    `hist` contient des cumuls INCLUANT leur propre date ; la jointure exclut la date de
    la course (allow_exact_matches=False) : aucune information du jour même n'est utilisée.
    """
    left = df.assign(_d=pd.to_datetime(df["race_date"])).sort_values("_d")
    right = hist.assign(_d=pd.to_datetime(hist["race_date"]),
                        _hist_date=pd.to_datetime(hist["race_date"]))
    right = right.drop(columns="race_date").sort_values("_d")
    for k in keys:  # merge_asof exige des types de clés identiques
        left[k] = left[k].astype("float64")
        right[k] = right[k].astype("float64")
    merged = pd.merge_asof(left, right, on="_d", by=keys, direction="backward",
                           allow_exact_matches=False)
    return merged.drop(columns="_d")


def _actor_history(hist: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    d = hist.dropna(subset=[key]).groupby([key, "race_date"]).agg(
        runs=("won", "size"), wins=("won", "sum")).reset_index().sort_values("race_date")
    g = d.groupby(key)
    d[f"{prefix}_cum_runs"] = g["runs"].cumsum()
    d[f"{prefix}_cum_wins"] = g["wins"].cumsum()
    return d[[key, "race_date", f"{prefix}_cum_runs", f"{prefix}_cum_wins"]]


def _horse_history(hist: pd.DataFrame) -> pd.DataFrame:
    d = hist.groupby(["horse_id", "race_date"]).agg(
        runs=("won", "size"), pct=("finish_pct", "mean"),
        horse_last_jockey=("jockey_id", "last")).reset_index()
    d = d.sort_values(["horse_id", "race_date"])
    g = d.groupby("horse_id")
    d["horse_db_runs"] = g["runs"].cumsum()
    d["horse_db_mean_pct"] = g["pct"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return d[["horse_id", "race_date", "horse_db_runs", "horse_db_mean_pct",
              "horse_last_jockey"]]


def _rolling_by(d: pd.DataFrame, key: str, col: str, window: int, how: str) -> pd.Series:
    r = d.groupby(key)[col].rolling(window, min_periods=1)
    out = r.mean() if how == "mean" else r.min()
    return out.reset_index(level=0, drop=True)


def _join_back(df: pd.DataFrame, looked: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rattache à `df` des colonnes calculées sur une copie réordonnée (clé de partant)."""
    return df.drop(columns=[c for c in cols if c in df.columns]).merge(
        looked[["race_id", "num_pmu", *cols]], on=["race_id", "num_pmu"], how="left")


def _speed(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Vitesse en trot : réduction kilométrique (s/km, plus petit = plus rapide) des courses
    passées, brute et relative à la médiane des arrivants de chaque course."""
    t = hist[hist["discipline"].astype(str).str.startswith("trot")
             & hist["reduction_km_s"].notna()]
    t = t.assign(rvf=t["reduction_km_s"]
                 - t.groupby("race_id")["reduction_km_s"].transform("median"))
    d = t.groupby(["horse_id", "race_date"]).agg(
        red=("reduction_km_s", "min"), rvf=("rvf", "min")).reset_index()
    d = d.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
    d["speed_red_mean3"] = _rolling_by(d, "horse_id", "red", 3, "mean")
    d["speed_red_best5"] = _rolling_by(d, "horse_id", "red", 5, "min")
    d["speed_vs_field_mean3"] = _rolling_by(d, "horse_id", "rvf", 3, "mean")
    d["speed_runs"] = d.groupby("horse_id").cumcount() + 1.0
    cols = ["speed_red_mean3", "speed_red_best5", "speed_vs_field_mean3", "speed_runs"]
    looked = _asof_before(df[["race_id", "num_pmu", "horse_id", "race_date"]],
                          d[["horse_id", "race_date", *cols]], ["horse_id"])
    df = _join_back(df, looked, cols)
    trot = df["discipline"].astype(str).str.startswith("trot")
    df["speed_runs"] = df["speed_runs"].where(~trot, df["speed_runs"].fillna(0.0))
    for c in ("speed_red_mean3", "speed_vs_field_mean3"):
        df[f"{c}_rel"] = df[c] - df.groupby("race_id")[c].transform("mean")
    return df


def _gear(df: pd.DataFrame) -> pd.DataFrame:
    """Ferrure et œillères du jour, et changement depuis la dernière course connue."""
    shoes = df["shoes"].astype(object).where(df["shoes"].notna(), "FERRE").astype(str)
    blink = df["blinkers"].astype(object).where(df["blinkers"].notna(), "SANS").astype(str)
    trot = df["discipline"].astype(str).str.startswith("trot")
    prev = pd.DataFrame({"race_id": df["race_id"], "num_pmu": df["num_pmu"],
                         "horse_id": df["horse_id"], "race_date": df["race_date"],
                         "prev_shoes": shoes, "prev_blinkers": blink})
    last = prev.sort_values(["horse_id", "race_date", "race_id"]).groupby(
        ["horse_id", "race_date"])[["prev_shoes", "prev_blinkers"]].last().reset_index()
    looked = _asof_before(prev[["race_id", "num_pmu", "horse_id", "race_date"]], last,
                          ["horse_id"])
    df = _join_back(df, looked, ["prev_shoes", "prev_blinkers"])
    ps, pb = df.pop("prev_shoes"), df.pop("prev_blinkers")
    d4 = df["shoes"].astype(str).eq(D4)
    deferre = df["shoes"].astype(str).str.contains("DEFERR", na=False)
    cur_shoes = df["shoes"].astype(object).where(df["shoes"].notna(), "FERRE").astype(str)
    cur_blink = df["blinkers"].astype(object).where(df["blinkers"].notna(), "SANS").astype(str)
    nan = np.nan
    df["shoes_d4"] = np.where(trot, d4.astype(float), nan)
    df["shoes_deferre"] = np.where(trot, deferre.astype(float), nan)
    df["shoes_change"] = np.where(trot & ps.notna(), (cur_shoes != ps).astype(float), nan)
    df["shoes_d4_first"] = np.where(trot & ps.notna(), (d4 & ps.ne(D4)).astype(float), nan)
    df["blinkers_on"] = (cur_blink != "SANS").astype(float)
    df["blinkers_change"] = np.where(pb.notna(), (cur_blink != pb).astype(float), nan)
    return df


def _keyed_record(df: pd.DataFrame, hist: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Nombre de courses passées du cheval avec la même valeur de `key` et rang d'arrivée
    moyen (percentile, 0 = vainqueur) lissé vers 0,5."""
    h = hist.dropna(subset=[key])
    d = h.groupby(["horse_id", key, "race_date"]).agg(
        r=("finish_pct", "size"), s=("finish_pct", "sum")).reset_index()
    d = d.sort_values("race_date")
    g = d.groupby(["horse_id", key])
    d[f"{prefix}_runs"], d["_s"] = g["r"].cumsum(), g["s"].cumsum()
    looked = _asof_before(df[["race_id", "num_pmu", "horse_id", key, "race_date"]].dropna(
        subset=[key]), d[["horse_id", key, "race_date", f"{prefix}_runs", "_s"]],
        ["horse_id", key])
    df = _join_back(df, looked, [f"{prefix}_runs", "_s"])
    runs = df[f"{prefix}_runs"].fillna(0.0)
    df[f"{prefix}_runs"] = runs.where(df[key].notna())
    df[f"{prefix}_pct"] = ((df.pop("_s").fillna(0.0) + 0.5 * APT_SHRINK)
                           / (runs + APT_SHRINK)).where(df[key].notna())
    return df


def _recent_form(df: pd.DataFrame, hist: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Montes / partants et victoires sur les RECENT_DAYS jours précédant la course
    (veille comprise, jour même exclu)."""
    d = hist.dropna(subset=[key]).groupby([key, "race_date"]).agg(
        r=("won", "size"), w=("won", "sum")).reset_index().sort_values("race_date")
    g = d.groupby(key)
    d["_cr"], d["_cw"] = g["r"].cumsum(), g["w"].cumsum()
    base = df[["race_id", "num_pmu", key, "race_date"]].dropna(subset=[key])
    now = _asof_before(base, d[[key, "race_date", "_cr", "_cw"]], [key])
    past = base.assign(race_date=pd.to_datetime(base["race_date"])
                       - pd.Timedelta(days=RECENT_DAYS))
    past["race_date"] = past["race_date"].dt.date
    then = _asof_before(past, d[[key, "race_date", "_cr", "_cw"]], [key]).rename(
        columns={"_cr": "_cr0", "_cw": "_cw0"})
    looked = now[["race_id", "num_pmu", "_cr", "_cw"]].merge(
        then[["race_id", "num_pmu", "_cr0", "_cw0"]], on=["race_id", "num_pmu"])
    looked[f"{prefix}_runs_30d"] = looked["_cr"].fillna(0) - looked["_cr0"].fillna(0)
    wins = looked["_cw"].fillna(0) - looked["_cw0"].fillna(0)
    looked[f"{prefix}_win_rate_30d"] = (wins + RECENT_SHRINK * PRIOR_WIN_RATE) / (
        looked[f"{prefix}_runs_30d"] + RECENT_SHRINK)
    return _join_back(df, looked, [f"{prefix}_runs_30d", f"{prefix}_win_rate_30d"])


def _add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables v2, toutes calculées sur des dates STRICTEMENT antérieures (sauf ferrure et
    œillères du jour, champs déclaratifs de l'engagement)."""
    trot = df["discipline"].astype(str).str.startswith("trot")
    df["_rc"] = pd.factorize(df["racecourse"].astype(object))[0].astype(float)
    df.loc[df["racecourse"].isna(), "_rc"] = np.nan
    disc_code = pd.factorize(df["discipline"].astype(object))[0]
    df["_dist"] = disc_code * 100.0 + np.round(df["distance_m"] / 400.0)
    df["_start"] = np.where(trot, df["autostart"].astype(float), np.nan)
    hist = df[df["has_result"]]
    df = _speed(df, hist)
    df = _gear(df)
    for key, prefix in (("_rc", "apt_course"), ("_dist", "apt_dist"), ("_start", "apt_start")):
        df = _keyed_record(df, hist, key, prefix)
    for key, prefix in (("jockey_id", "jockey"), ("trainer_id", "trainer")):
        df = _recent_form(df, hist, key, prefix)
    return df.drop(columns=["_rc", "_dist", "_start"])


GOING_LEVELS = (("très léger", 0), ("léger", 0), ("bon léger", 1), ("bon souple", 2),
                ("très souple", 3), ("souple", 2), ("bon", 1), ("collant", 3),
                ("très lourd", 4), ("lourd", 4))


def _going_level(label: object, pen: float) -> float:
    """Terrain ordonné : 0 léger, 1 bon, 2 souple, 3 très souple / collant, 4 lourd."""
    text = str(label or "").strip().lower()
    for key, level in GOING_LEVELS:  # les libellés composés sont testés en premier
        if text.startswith(key):
            return float(level)
    if np.isfinite(pen):
        return float(np.digitize(pen, [2.8, 3.25, 3.75, 4.55]))
    return np.nan


def _going(going: pd.DataFrame | None) -> pd.DataFrame:
    """Terrain annoncé de chaque course (première observation : condition d'avant-course)."""
    cols = ["race_id", "going_pen", "going_level", "going_psf"]
    if going is None or going.empty:
        return pd.DataFrame(columns=cols)
    g = going.sort_values("observed_at").drop_duplicates("race_id", keep="first").copy()
    g["going_pen"] = pd.to_numeric(g["penetrometer"].astype(str).str.replace(",", "."),
                                   errors="coerce")
    g["going_psf"] = g["going_label"].astype(str).str.upper().str.startswith("PSF").astype(
        float)
    g["going_level"] = [np.nan if psf else _going_level(lab, pen) for lab, pen, psf in
                        zip(g["going_label"], g["going_pen"], g["going_psf"], strict=True)]
    return g[cols]


def _actor_keyed_rate(df: pd.DataFrame, hist: pd.DataFrame, keys: list[str],
                      prefix: str) -> pd.DataFrame:
    """Partants et victoires passés d'un acteur dans un contexte (ex. entraîneur × hippodrome),
    lissés vers le taux a priori."""
    d = hist.dropna(subset=keys).groupby([*keys, "race_date"]).agg(
        r=("won", "size"), w=("won", "sum")).reset_index().sort_values("race_date")
    g = d.groupby(keys)
    d[f"{prefix}_runs"], d["_w"] = g["r"].cumsum(), g["w"].cumsum()
    looked = _asof_before(df[["race_id", "num_pmu", *keys, "race_date"]].dropna(subset=keys),
                          d[[*keys, "race_date", f"{prefix}_runs", "_w"]], keys)
    df = _join_back(df, looked, [f"{prefix}_runs", "_w"])
    runs = df[f"{prefix}_runs"].fillna(0.0)
    df[f"{prefix}_runs"] = runs
    df[f"{prefix}_win_rate"] = (df.pop("_w").fillna(0.0) + SHRINK_RUNS * PRIOR_WIN_RATE) / (
        runs + SHRINK_RUNS)
    return df


def _previous_race(df: pd.DataFrame) -> pd.DataFrame:
    """Dotation et distance des courses précédentes du cheval (dates strictement
    antérieures) : changement de catégorie et de distance."""
    d = df[["horse_id", "race_date", "log_prize", "distance_m"]].groupby(
        ["horse_id", "race_date"]).agg(lp=("log_prize", "mean"),
                                        dist=("distance_m", "last")).reset_index()
    d = d.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
    d["_lp3"] = _rolling_by(d, "horse_id", "lp", 3, "mean")
    d = d.rename(columns={"dist": "_prev_dist"})
    looked = _asof_before(df[["race_id", "num_pmu", "horse_id", "race_date"]],
                          d[["horse_id", "race_date", "_lp3", "_prev_dist"]], ["horse_id"])
    df = _join_back(df, looked, ["_lp3", "_prev_dist"])
    df["class_change"] = df["log_prize"] - df.pop("_lp3")
    df["dist_change"] = df["distance_m"] - df.pop("_prev_dist")
    return df


def _add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables v3 (dates strictement antérieures, sauf le terrain annoncé du jour)."""
    df["_rc"] = pd.factorize(df["racecourse"].astype(object))[0].astype(float)
    df.loc[df["racecourse"].isna(), "_rc"] = np.nan
    df["_going"] = np.where(df["going_psf"] == 1, 9.0, df["going_level"])
    hist = df[df["has_result"]]
    df = _keyed_record(df, hist, "_going", "apt_going")
    df = _keyed_record(df, hist, "jockey_id", "apt_pair")
    df = _actor_keyed_rate(df, hist, ["trainer_id", "_rc"], "trainer_course")
    df = _previous_race(df)
    return df.drop(columns=["_rc", "_going"])


def build_dataset(raw: dict[str, pd.DataFrame], use_source_aggregates: bool = True) -> Dataset:
    races = raw["races"].copy()
    races["decision_time"] = races["scheduled_start_utc"] - DECISION_OFFSET
    races = races.merge(_going(raw.get("going")), on="race_id", how="left")
    for c in ("going_pen", "going_level", "going_psf"):
        races[c] = pd.to_numeric(races[c], errors="coerce").astype("float64")
    races = races[races["scheduled_start_utc"].notna()]

    e = _entries_at_decision(raw["entries"], races)
    e = e[e["status"].fillna("").str.upper() != "NON_PARTANT"]
    df = e.merge(races, on=["race_id", "decision_time"])
    df = df.merge(raw["results"], on=["race_id", "num_pmu"], how="left")

    n = df.groupby("race_id")["num_pmu"].transform("size")
    df["n_runners"] = n.astype(float)
    df["won"] = (df["finish_position"] == 1).astype(float)
    df["has_result"] = df.groupby("race_id")["finish_position"].transform(
        lambda s: s.notna().any()) & df["result_final"].astype(bool)
    # rang d'arrivée en percentile (0 = premier, 1 = dernier ; non classé = 1)
    df["finish_pct"] = ((df["finish_position"] - 1) / (n - 1).clip(lower=1)).clip(0, 1)
    df.loc[df["finish_position"].isna(), "finish_pct"] = 1.0
    df.loc[~df["has_result"], ["won", "finish_pct"]] = np.nan

    # --- conditions de course et engagement
    df["draw_rel"] = df["draw"] / n
    df["log_prize"] = np.log1p(df["prize_eur"])
    df["weight_vs_field"] = df["weight_carried_kg"] - df.groupby("race_id")[
        "weight_carried_kg"].transform("mean")
    df["handicap_extra_m"] = df["handicap_distance_m"] - df["distance_m"]

    # --- agrégats de la source
    mus = pd.DataFrame([musique_features(f, d) for f, d in
                        zip(df["form_string"], df["discipline"], strict=True)], index=df.index)
    df = pd.concat([df, mus], axis=1)
    starts = df["career_starts"]
    df["career_win_rate"] = (df["career_wins"] + 0.1 * CAREER_SHRINK) / (starts + CAREER_SHRINK)
    df["career_place_rate"] = (df["career_places"] + 0.3 * CAREER_SHRINK) / (
        starts + CAREER_SHRINK)
    df["log_earnings_per_start"] = np.log1p(df["career_earnings"] / starts.where(starts > 0))
    if not use_source_aggregates:
        source_cols = [*mus.columns, "career_starts", "career_win_rate", "career_place_rate",
                       "log_earnings_per_start"]
        df[source_cols] = np.nan

    # --- historique interne (dates strictement antérieures)
    hist = df[df["has_result"]]
    prior = PRIOR_WIN_RATE
    for key, prefix in (("jockey_id", "jockey"), ("trainer_id", "trainer")):
        df = _asof_before(df, _actor_history(hist, key, prefix), [key]).drop(
            columns="_hist_date")
        runs = df.pop(f"{prefix}_cum_runs").fillna(0.0)
        wins = df.pop(f"{prefix}_cum_wins").fillna(0.0)
        df[f"{prefix}_db_runs"] = runs
        df[f"{prefix}_win_rate"] = (wins + SHRINK_RUNS * prior) / (runs + SHRINK_RUNS)
    df = _asof_before(df, _horse_history(hist), ["horse_id"])
    df["horse_days_since_db"] = (pd.to_datetime(df["race_date"]) - df.pop("_hist_date")).dt.days
    df["horse_db_runs"] = df["horse_db_runs"].fillna(0.0)
    # changement de jockey/driver depuis la dernière course connue (NaN sans historique)
    last = df.pop("horse_last_jockey")
    df["rider_change"] = np.where(last.isna(), np.nan,
                                  (df["jockey_id"].astype("float64") != last).astype(float))
    pair = hist.groupby(["horse_id", "jockey_id", "race_date"]).size().rename("r").reset_index()
    pair = pair.sort_values("race_date")
    pair["pair_db_runs"] = pair.groupby(["horse_id", "jockey_id"])["r"].cumsum()
    df = _asof_before(df, pair[["horse_id", "jockey_id", "race_date", "pair_db_runs"]],
                      ["horse_id", "jockey_id"]).drop(columns="_hist_date")
    df["pair_db_runs"] = df["pair_db_runs"].fillna(0.0)

    df = _add_v2_features(df)
    df = _add_v3_features(df)

    # --- variables relatives à la course
    for c in RELATIVE_SOURCES:
        df[f"{c}_rel"] = df[c] - df.groupby("race_id")[c].transform("mean")

    # --- marché
    df = df.merge(_market(raw["odds"], races, df), on=["race_id", "num_pmu"], how="left")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return Dataset(df.sort_values(["race_date", "race_id", "num_pmu"]).reset_index(drop=True))
