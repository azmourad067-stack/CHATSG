"""Sélection du jour : deux règles NON validées, appliquées telles qu'elles ont été définies,
et la règle « déferré des 4 pour la 1re fois, autostart, place 4 à 6, cote ≤ 9 » figée le
23/11/2026 (evaluation/forward_rules.D4_AUTOSTART), qui remplace l'onglet d'information
« déferré des 4 (1re fois) » du même jour, à la demande de l'utilisateur.

1. « 3e du modèle, simple gagnant, trot G » (ci-dessous) ; toutes réunions ouvertes aux
   paris en ligne depuis le 23/11/2026, à la demande de l'utilisateur (test du même jour :
   −3,2 % en nationales, −2,5 % en régionales, −2,8 % au total sur deux ans).
2. « Valeur 4e-6e du modèle, cote 9 à 20, 10 à 14 partants, espérance ≥ +10 %, gagnant »,
   figée le 21/11/2026 (evaluation/forward_rules.py) à la demande de l'utilisateur : seule
   famille positive sur les trois périodes de la grande recherche, mais NON significative.
   Toutes réunions ouvertes aux paris en ligne (cotes Internet), comme dans la recherche.
   Le profil « outsiders attelé », ajouté puis retiré le 21/11/2026, n'est plus affiché :
   −70 % sur les six derniers mois de la grande recherche.

Règle 1 :

Règle « 3e du modèle, simple gagnant, trot G » : la seule des règles figées le 10/11/2026
restée positive sur l'année de validation avec le modèle d'origine (+6,0 %, IC 95 %
[−24 % ; +38 %]). Elle n'est PAS validée : réentraîné, le même modèle donne −1,2 % sur
cette année. Avec le modèle actuel (15/11/2026), sur les seules réunions NATIONALES (les
seules affichées ici), elle donne −11,7 % sur deux ans (407 paris, IC [−35 % ; +15 %]).
La page l'applique telle qu'elle a été définie, pour un suivi fictif ou à petites mises
fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import select

from horse_intelligence.data.storage import database as db
from horse_intelligence.evaluation.forward_rules import (
    D4_AUTOSTART,
    SMALL_VALUE,
    VALUE_46,
    d4_rule_match,
    is_autostart,
    is_small_race,
    races_with_internet_odds,
    rule_picks,
    start_place,
)
from horse_intelligence.evaluation.segment_analysis import race_group
from horse_intelligence.frontend.components.cards import notice
from horse_intelligence.frontend.race_analysis import day_predictions
from horse_intelligence.frontend.state import get_context
from horse_intelligence.prediction.signals import first_time_d4

PARIS = ZoneInfo("Europe/Paris")
RULE_GROUP = "Trot G"
RULE_RANK = 3  # 3e cheval du modèle
RULE_MODEL = "lgbm_market"
# réunions régionales et locales (audience de l'API PMU)
REGIONAL_AUDIENCES = ("LOCAL", "REGIONAL")



def national_races(races: pd.DataFrame) -> pd.DataFrame:
    """Courses des réunions du programme national (audience inconnue : conservée)."""
    if races.empty or "audience" not in races:
        return races
    return races[~races["audience"].isin(REGIONAL_AUDIENCES)]


def rule_races(races: pd.DataFrame) -> pd.DataFrame:
    """Courses de la catégorie visée, classées comme dans les tests (libellé + conditions)."""
    if races.empty:
        return races
    groups = [race_group(lab, cond, d) for lab, cond, d in
              zip(races["label"], races["conditions_text"], races["discipline"], strict=True)]
    return races[[g == RULE_GROUP for g in groups]]


def rule_pick(table: pd.DataFrame) -> pd.Series | None:
    """Cheval classé 3e par le modèle (à égalité, l'ordre du tableau départage)."""
    if len(table) < RULE_RANK:
        return None
    return table.sort_values("p_win", ascending=False, kind="stable").iloc[RULE_RANK - 1]


def _races_on(day: date) -> pd.DataFrame:
    r, rc = db.races, db.racecourses
    stmt = (select(r.c.race_id, r.c.reunion_num, r.c.course_num, r.c.label,
                   r.c.conditions_text, r.c.discipline, r.c.distance_m,
                   r.c.scheduled_start_utc, r.c.result_final, r.c.audience,
                   r.c.prize_eur, rc.c.name.label("racecourse"))
            .join(rc, rc.c.racecourse_id == r.c.racecourse_id, isouter=True)
            .where(r.c.race_date == day).order_by(r.c.scheduled_start_utc))
    with get_context().repo.engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def _paris_time(value: object) -> str:
    if value is None or pd.isna(value):  # type: ignore[call-overload]
        return "—"
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts
    return ts.tz_convert(PARIS).strftime("%H:%M")


def _finish(race: dict[Any, Any], num_pmu: Any) -> Any:
    if not bool(race["result_final"]):
        return None
    res = get_context().repo.results_for(int(race["race_id"]))
    hit = res[res["num_pmu"] == num_pmu]
    return None if hit.empty else hit["finish_position"].iloc[0]


@dataclass
class DayData:
    """Prédictions et partants des courses du jour, lus une seule fois par affichage."""

    preds: dict[int, Any]
    entries: dict[int, pd.DataFrame]
    online: set[int] = field(default_factory=set)  # cotes Internet publiées
    d4_first: set[tuple[int, int]] = field(default_factory=set)  # déferrés 4 1re fois

    def pred(self, race_id: int) -> Any:
        return self.preds.get(race_id)

    def runners(self, race_id: int) -> pd.DataFrame:
        return self.entries.get(race_id, pd.DataFrame())

    def names(self, race_id: int) -> dict[int, str]:
        e = self.runners(race_id)
        return dict(zip(e["horse_id"], e["horse"], strict=True)) if not e.empty else {}


def _day_data(day: date, races: pd.DataFrame) -> DayData:
    ids = [int(x) for x in races["race_id"]] if not races.empty else []
    if not ids:
        return DayData({}, {})
    with st.spinner("Calcul des probabilités…"):
        preds = day_predictions(day, ids)
        repo = get_context().repo
        entries = repo.entries_for_races(ids)
        online = races_with_internet_odds(repo.engine, ids)
        try:  # signal d'information : ne doit jamais empêcher la page de s'afficher
            d4 = first_time_d4(repo.engine, ids, day)
        except Exception:
            d4 = set()
    by_race = {int(cast(int, rid)): g.reset_index(drop=True)
               for rid, g in entries.groupby("race_id")} if not entries.empty else {}
    return DayData(preds, by_race, online, d4)


def render() -> None:
    st.title("Sélection du jour")
    repo = get_context().repo
    dates = repo.available_dates()
    if not dates:
        notice("Aucune course en base.")
        return
    today = datetime.now(PARIS).date()
    day = st.date_input("Date", value=today if today in dates else dates[0],
                        min_value=dates[-1], max_value=dates[0], format="DD/MM/YYYY",
                        key="selection_date")
    day = day if isinstance(day, date) else dates[0]
    all_races = _races_on(day)
    data = _day_data(day, all_races)
    tab_g, tab_value, tab_small, tab_d4 = st.tabs([
        "3e du modèle, trot G", "Valeur 4e-6e (cote 9-20)", "Petites courses (valeur)",
        "Déferré 4 autostart (place 4-6)"])
    with tab_g:
        _render_trot_g(all_races, data)
    with tab_value:
        _render_value(all_races, data)
    with tab_small:
        _render_small(all_races, data)
    with tab_d4:
        _render_d4(all_races, data)
    st.caption(
        "Heures de Paris. Le classement tient compte des cotes, il peut changer jusqu'au "
        "départ : consultez la page 10 à 20 minutes avant la course (les données sont "
        "rafraîchies toutes les 15 minutes) et jouez le cheval affiché à ce moment, sans "
        "revenir dessus. Les tests ont été réglés aux rapports de la masse Internet du PMU.")


def _render_trot_g(all_races: pd.DataFrame, data: DayData) -> None:
    st.warning("Règle NON validée : « 3e du modèle, simple gagnant, courses de trot G », "
               "toutes réunions (nationales, régionales, locales) ouvertes aux paris en "
               "ligne. Avec le modèle actuel (réentraîné le 22/11/2026), elle aurait donné "
               "−2,8 % sur deux ans (871 paris, intervalle de −21 % à +16 %) : aucun "
               "avantage démontré. À suivre de façon fictive, ou pour le plaisir avec de "
               "petites mises fixes.")
    races = rule_races(all_races)
    if races.empty:
        notice("Aucune course de trot G ce jour-là : rien à jouer.")
        return

    rows = []
    for r in races.to_dict("records"):
        rid = int(r["race_id"])
        pred = data.pred(rid)
        # règle testée avec le modèle « LightGBM + marché » : sans cotes, pas de choix ;
        # jouée seulement là où le pari en ligne est ouvert (cotes Internet publiées)
        tested = pred is not None and pred.primary == RULE_MODEL and not pred.table.empty
        online = rid in data.online
        pick = rule_pick(pred.table) if tested and online else None
        names = data.names(rid)
        finish = None if pick is None else _finish(r, pick["num_pmu"])
        if pick is not None:
            remark = ""
        elif tested:
            remark = "pas de pari en ligne (aucune cote Internet)"
        elif pred is not None and not pred.table.empty:
            remark = "cotes pas encore publiées : revenir plus tard"
        else:
            remark = pred.message if pred is not None else "aucune prédiction"
        rows.append({
            "Départ": _paris_time(r["scheduled_start_utc"]),
            "Course": f"R{r['reunion_num']}C{r['course_num']}",
            "Hippodrome": r["racecourse"] or "?",
            "Réunion": AUDIENCE_NAMES.get(str(r.get("audience")), "nationale"),
            "Distance": f"{r['distance_m'] or '?'} m",
            "N° à jouer": "—" if pick is None else str(int(pick["num_pmu"])),
            "Cheval": "—" if pick is None else names.get(pick["horse_id"], "?"),
            "P(victoire)": "—" if pick is None else f"{100 * pick['p_win']:.0f} %",
            "Cote réf.": "—" if pick is None or pd.isna(pick["mkt_odds"])
            else f"{pick['mkt_odds']:.1f}",
            "Arrivée": "—" if finish is None or pd.isna(finish) else str(int(finish)),
            "Remarque": remark,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


DISCIPLINE_NAMES = {"trot_attele": "trot attelé", "trot_monte": "trot monté", "plat": "plat",
                    "obstacle": "obstacle"}
CHECK_BEFORE = timedelta(minutes=10)


def n_starters(entries: pd.DataFrame) -> int:
    """Partants connus d'une course (non-partants déclarés exclus)."""
    if entries.empty:
        return 0
    status = entries["status"] if "status" in entries else pd.Series("", index=entries.index)
    return int((status.fillna("").str.upper() != "NON_PARTANT").sum())


def watch_state(table: pd.DataFrame, picks: pd.DataFrame, started: bool,
                finished: bool) -> str:
    """État d'une course à surveiller, en clair."""
    if finished:
        return "terminée"
    if table.empty or table["mkt_odds"].isna().any():
        return "partie" if started else "cotes pas encore publiées : revenir avant le départ"
    if not picks.empty:
        nums = ", ".join(f"n°{int(n)}" for n in picks["num_pmu"])
        return f"{len(picks)} cheval(aux) retenu(s) pour l'instant : {nums}"
    return "aucun cheval retenu pour l'instant"


AUDIENCE_NAMES = {"NATIONAL": "nationale", "REGIONAL": "régionale", "LOCAL": "locale"}


def _render_value(races: pd.DataFrame, data: DayData) -> None:
    rule = VALUE_46
    st.warning(
        "Règle NON validée, figée le 21/11/2026 pour être jugée sur les courses à venir : "
        "« 4e à 6e du modèle, cote 9 à 20, 10 à 14 partants, espérance du modèle d'au moins "
        "+10 %, simple gagnant ». Sur deux ans de tests elle a donné +22,6 %, puis +41,8 %, "
        "puis +22,0 % sur les six derniers mois, mais avec une marge d'erreur de −29 % à "
        "+76 % : c'est peut-être de la chance (8 640 stratégies essayées). Environ 800 "
        "paris par an (toutes réunions ouvertes aux paris en ligne), environ 1 gagnant sur "
        "9. À suivre de façon fictive, ou avec de très petites mises fixes.")
    st.info(
        "Seul le SMS fait foi : il est envoyé 45 à 60 minutes avant le départ, avec les "
        "chevaux retenus à ce moment-là (version « 1 h avant » choisie le 23/11/2026). Cet "
        "onglet montre, lui, le choix du moment : quand le PMU publie ses cotes de "
        "référence, 30 minutes avant le départ, un cheval peut entrer ou sortir de la "
        "liste. Un cheval apparu ici après l'heure du SMS est hors règle : ne le jouez pas.")
    if races.empty:
        notice("Aucune course ce jour-là.")
        return
    now = datetime.now(UTC)
    rows, watch = [], []
    empty = pd.DataFrame(columns=["num_pmu", "horse_id", "p_win", "mkt_odds"])
    for r in races.to_dict("records"):
        rid = int(r["race_id"])
        pred = data.pred(rid)
        priced = pred is not None and pred.primary == RULE_MODEL
        online = rid in data.online
        # la règle n'est jouée que là où le pari en ligne est ouvert (cotes Internet)
        table = pred.table if priced and online else empty
        entries = data.runners(rid)
        n = len(table) if not table.empty else n_starters(entries)
        if not rule.runners[0] <= n <= rule.runners[1]:
            continue
        picks = rule_picks(table, rule) if not table.empty else table
        start = r["scheduled_start_utc"]
        start_ts = None if start is None or pd.isna(start) else pd.Timestamp(start)
        if start_ts is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(UTC)
        started = start_ts is not None and start_ts <= now
        watch.append({
            "_start": start_ts,
            "Départ": _paris_time(start),
            "Consulter vers": "—" if start_ts is None
            else _paris_time(start_ts - CHECK_BEFORE),
            "Course": f"R{r['reunion_num']}C{r['course_num']}",
            "Hippodrome": r["racecourse"] or "?",
            "Réunion": AUDIENCE_NAMES.get(str(r.get("audience")), "nationale"),
            "Discipline": DISCIPLINE_NAMES.get(str(r["discipline"]), "?"),
            "Partants": n,
            "État": "pas de pari en ligne (aucune cote Internet)"
            if priced and not online and not bool(r["result_final"])
            else watch_state(table, picks, started, bool(r["result_final"])),
        })
        if picks.empty:
            continue
        names = dict(zip(entries["horse_id"], entries["horse"], strict=True)) \
            if not entries.empty else {}
        for pick in picks.to_dict("records"):
            finish = _finish(r, int(pick["num_pmu"]))
            rows.append({
                "Départ": _paris_time(start),
                "Course": f"R{r['reunion_num']}C{r['course_num']}",
                "Hippodrome": r["racecourse"] or "?",
                "Partants": n,
                "N° à jouer": int(pick["num_pmu"]),
                "Cheval": names.get(pick["horse_id"], "?"),
                "Rang modèle": int(pick["model_rank"]),
                "P(victoire)": f"{100 * pick['p_win']:.1f} %",
                "Cote réf.": f"{pick['mkt_odds']:.1f}",
                "Espérance": f"{100 * pick['ev']:+.0f} %",
                "Arrivée": "—" if finish is None or pd.isna(finish) else int(finish),
            })
    if not watch:
        notice("Aucune course de 10 à 14 partants ce jour-là.")
        return
    watch_df = pd.DataFrame(watch).sort_values("_start", na_position="last")
    upcoming = watch_df[watch_df["_start"].notna() & (watch_df["_start"] > now)]
    if not upcoming.empty:
        nxt = upcoming.iloc[0]
        st.info(f"Prochaine course à surveiller : {nxt['Course']} ({nxt['Hippodrome']}) "
                f"à {nxt['Départ']}. Rafraîchissez la page vers {nxt['Consulter vers']}.")
    st.subheader("Chevaux retenus")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        notice("Aucun cheval retenu pour l'instant. La liste change avec les cotes : "
               "revenez 10 minutes avant chaque course ci-dessous.")
    st.subheader(f"Courses à surveiller ({len(watch_df)})")
    st.dataframe(watch_df.drop(columns="_start"), hide_index=True, width="stretch")
    notes = ["Courses de 10 à 14 partants, toutes réunions (nationales, régionales, "
             "locales) : la règle ne peut retenir un cheval que dans celles-ci, et seulement "
             "si le pari en ligne y est ouvert (cotes Internet publiées). Le choix dépend des "
             "cotes, qui bougent jusqu'au départ : rafraîchissez la page à l'heure « Consulter "
             "vers » et jouez les chevaux retenus à ce moment-là, sans revenir dessus."]
    st.caption(" ".join(notes) + " Une même course peut donner plusieurs chevaux (1 € "
               "chacun dans les tests). L'espérance est calculée sur la cote de référence "
               "× 0,90 : ce n'est pas un gain garanti.")


# décision de la règle « petites courses » : 13 à 28 min avant le départ, une fois publiées
# les cotes de référence du PMU (30 min avant), comme dans son test
SMALL_CHECK = (timedelta(minutes=28), timedelta(minutes=13))


def _start_ts(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):  # type: ignore[call-overload]
        return None
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    return ts.tz_localize(UTC) if ts.tzinfo is None else ts


def _render_small(races: pd.DataFrame, data: DayData) -> None:
    rule = SMALL_VALUE
    st.warning(
        "Règle NON validée, figée le 24/11/2026 pour être jugée sur les courses à venir : "
        "« petites courses (réunions régionales ou locales, ou allocation jusqu'à 10 000 €), "
        "espérance du modèle d'au moins +10 %, cote 2 à 20, simple gagnant ». Fixée avant "
        "son test sur deux ans : +15,4 % (1 535 paris, environ 1 gagnant sur 4), +8,0 % sur "
        "les six derniers mois ; marge d'erreur de −2 % à +35 % : pas encore une preuve. À "
        "suivre de façon fictive, ou avec de très petites mises fixes.")
    small = races[[is_small_race(a, z) for a, z in
                   zip(races.get("audience", pd.Series(dtype=object)),
                       races.get("prize_eur", pd.Series(dtype=float)), strict=True)]] \
        if not races.empty else races
    if small.empty:
        notice("Aucune petite course ce jour-là (réunion régionale ou locale, ou allocation "
               "jusqu'à 10 000 €).")
        return
    now = datetime.now(UTC)
    rows, watch = [], []
    for r in small.to_dict("records"):
        rid = int(r["race_id"])
        pred = data.pred(rid)
        priced = pred is not None and pred.primary == RULE_MODEL and not pred.table.empty
        online = rid in data.online
        table = pred.table if priced and online else pd.DataFrame(
            columns=["num_pmu", "horse_id", "p_win", "mkt_odds"])
        picks = rule_picks(table, rule) if not table.empty else table
        start_ts = _start_ts(r["scheduled_start_utc"])
        started = start_ts is not None and start_ts <= now
        finished = bool(r["result_final"])
        prize = pd.to_numeric(pd.Series([r.get("prize_eur")]), errors="coerce").iloc[0]
        watch.append({
            "_start": start_ts,
            "Départ": _paris_time(start_ts),
            "Décider entre": "—" if start_ts is None else
            f"{_paris_time(start_ts - SMALL_CHECK[0])} et {_paris_time(start_ts - SMALL_CHECK[1])}",
            "Course": f"R{r['reunion_num']}C{r['course_num']}",
            "Hippodrome": r["racecourse"] or "?",
            "Réunion": AUDIENCE_NAMES.get(str(r.get("audience")), "nationale"),
            "Allocation": "—" if pd.isna(prize) else f"{float(prize):,.0f} €".replace(",", " "),
            "État": "pas de pari en ligne (aucune cote Internet)"
            if pred is not None and pred.primary == RULE_MODEL and not online and not finished
            else watch_state(table, picks, started, finished),
        })
        names = data.names(rid)
        for pick in picks.to_dict("records"):
            finish = _finish(r, int(pick["num_pmu"]))
            rows.append({
                "Départ": _paris_time(start_ts),
                "Course": f"R{r['reunion_num']}C{r['course_num']}",
                "Hippodrome": r["racecourse"] or "?",
                "N° à jouer": int(pick["num_pmu"]),
                "Cheval": names.get(pick["horse_id"], "?"),
                "Rang modèle": int(pick["model_rank"]),
                "P(victoire)": f"{100 * pick['p_win']:.1f} %",
                "Cote réf.": f"{pick['mkt_odds']:.1f}",
                "Espérance": f"{100 * pick['ev']:+.0f} %",
                "Arrivée": "—" if finish is None or pd.isna(finish) else int(finish),
            })
    watch_df = pd.DataFrame(watch).sort_values("_start", na_position="last")
    upcoming = watch_df[watch_df["_start"].notna() & (watch_df["_start"] > now)]
    if not upcoming.empty:
        nxt = upcoming.iloc[0]
        st.info(f"Prochaine petite course : {nxt['Course']} ({nxt['Hippodrome']}) à "
                f"{nxt['Départ']}. Décision entre {nxt['Décider entre']} (un SMS est envoyé "
                f"à ce moment si un cheval est retenu).")
    st.subheader("Chevaux retenus")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        notice("Aucun cheval retenu pour l'instant. La liste dépend des cotes : elle n'est "
               "valable qu'entre 28 et 13 minutes avant le départ.")
    st.subheader(f"Petites courses du jour ({len(watch_df)})")
    st.dataframe(watch_df.drop(columns="_start"), hide_index=True, width="stretch")
    st.caption("Le test sur deux ans décide sur les cotes de référence que le PMU publie 30 "
               "minutes avant le départ : jouez les chevaux retenus entre 28 et 13 minutes "
               "avant la course, sans revenir dessus. Seules les courses ouvertes au pari en "
               "ligne (cotes Internet) sont jouées. L'espérance est calculée sur la cote × "
               "0,90 : ce n'est pas un gain garanti.")


def _render_d4(races: pd.DataFrame, data: DayData) -> None:
    rule = D4_AUTOSTART
    st.warning(
        "Règle NON validée, figée le 23/11/2026 pour être jugée sur les courses à venir : "
        "« trotteur déferré des 4 pour la 1re fois, départ autostart, place 4 à 6 derrière "
        "l'autostart, cote jusqu'à 9, simple gagnant ». Sur deux ans de tests : +16,4 %, puis "
        "+26,2 %, puis +17,7 % sur les six derniers mois, environ 1 gagnant sur 3 et 250 "
        "paris par an ; mais marge d'erreur de −27 % à +66 % : c'est peut-être de la chance "
        "(60 combinaisons essayées). À suivre de façon fictive, ou avec de très petites "
        "mises fixes.")
    if races.empty:
        notice("Aucune course ce jour-là.")
        return
    info = races.set_index("race_id")
    now = datetime.now(UTC)
    rows = []
    for rid, num in sorted(data.d4_first):
        if rid not in info.index:
            continue
        r = info.loc[rid]
        if not is_autostart(r.get("conditions_text")):
            continue
        e = data.runners(rid)
        hit = e[e["num_pmu"] == num] if not e.empty else e
        draw = hit["draw"].iloc[0] if not hit.empty and "draw" in hit else None
        place = start_place(draw, num)
        if not rule.places[0] <= place <= rule.places[1]:
            continue
        pred = data.pred(rid)
        odds, rank, p_win = None, "—", "—"
        if pred is not None and pred.primary == RULE_MODEL and not pred.table.empty:
            t = pred.table.sort_values("p_win", ascending=False, kind="stable")
            t = t.reset_index(drop=True)
            h = t[t["num_pmu"] == num]
            if not h.empty:
                odds = h["mkt_odds"].iloc[0]
                rank = str(int(h.index[0]) + 1)
                p_win = f"{100 * h['p_win'].iloc[0]:.1f} %"
        online = rid in data.online
        start = r["scheduled_start_utc"]
        start_ts = None if start is None or pd.isna(start) else pd.Timestamp(start)
        if start_ts is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(UTC)
        finished = bool(r["result_final"])
        if finished:
            state = "terminée"
        elif odds is None or pd.isna(odds):
            state = "cotes pas encore publiées : revenir avant le départ"
        elif not online:
            state = "pas de pari en ligne (aucune cote Internet)"
        elif d4_rule_match(place, odds, True, True, rule):
            state = "RETENU pour l'instant : à jouer en simple gagnant"
        else:
            state = f"cote {float(odds):.1f} au-dessus de {rule.odds[1]:g} : non retenu"
        finish = _finish({"race_id": rid, "result_final": r["result_final"]}, num)
        rows.append({
            "_start": str(start),
            "Départ": _paris_time(start),
            "Consulter vers": "—" if start_ts is None
            else _paris_time(start_ts - CHECK_BEFORE),
            "Course": f"R{r['reunion_num']}C{r['course_num']}",
            "Hippodrome": r["racecourse"] or "?",
            "Réunion": AUDIENCE_NAMES.get(str(r.get("audience")), "nationale"),
            "N°": num, "Cheval": data.names(rid).get(_horse_id(data, rid, num), "?"),
            "Place départ": str(int(place)),
            "Rang modèle": rank, "P(victoire)": p_win,
            "Cote réf.": "—" if odds is None or pd.isna(odds) else f"{float(odds):.1f}",
            "État": state,
            "Arrivée": "—" if finish is None or pd.isna(finish) else str(int(finish)),
            "_upcoming": start_ts is not None and start_ts > now and not finished,
        })
    if not rows:
        notice("Aucun trotteur déferré des 4 pour la 1re fois en place 4 à 6 d'un départ "
               "autostart ce jour-là (pour l'instant).")
        return
    table = pd.DataFrame(rows).sort_values(["_start", "Course", "N°"])
    upcoming = table[table["_upcoming"]]
    if not upcoming.empty:
        nxt = upcoming.iloc[0]
        st.info(f"Prochain cheval à surveiller : n°{nxt['N°']} {nxt['Cheval']}, "
                f"{nxt['Course']} ({nxt['Hippodrome']}) à {nxt['Départ']}. Rafraîchissez "
                f"la page vers {nxt['Consulter vers']} : il est retenu si sa cote est "
                f"jusqu'à {rule.odds[1]:g}.")
    st.dataframe(table.drop(columns=["_start", "_upcoming"]), hide_index=True,
                 width="stretch")
    st.caption("Candidats : trotteurs déferrés des 4 pieds alors qu'ils ne l'étaient pas à "
               "leur dernière course, partant en place 4 à 6 d'un départ autostart (numéro "
               "de corde publié, sinon numéro PMU). Le cheval est retenu si sa cote de "
               "référence est au plus 9 et que le pari en ligne est ouvert ; la cote bouge "
               "jusqu'au départ : décidez à l'heure « Consulter vers », sans revenir dessus.")


def _horse_id(data: DayData, race_id: int, num: int) -> Any:
    e = data.runners(race_id)
    hit = e[e["num_pmu"] == num] if not e.empty else e
    return None if hit.empty else hit["horse_id"].iloc[0]
