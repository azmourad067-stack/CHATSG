"""Sélection du jour : deux règles NON validées, appliquées telles qu'elles ont été définies.

1. « 3e du modèle, simple gagnant, trot G » (ci-dessous).
2. « Valeur 4e-6e du modèle, cote 9 à 20, 10 à 14 partants, espérance ≥ +10 %, gagnant »,
   figée le 21/11/2026 (evaluation/forward_rules.py) à la demande de l'utilisateur : seule
   famille positive sur les trois périodes de la grande recherche, mais NON significative.
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

from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import select

from horse_intelligence.data.storage import database as db
from horse_intelligence.evaluation.forward_rules import VALUE_46, rule_picks
from horse_intelligence.evaluation.segment_analysis import race_group
from horse_intelligence.frontend.components.cards import notice
from horse_intelligence.frontend.race_analysis import _predict
from horse_intelligence.frontend.state import get_context

PARIS = ZoneInfo("Europe/Paris")
RULE_GROUP = "Trot G"
RULE_RANK = 3  # 3e cheval du modèle
RULE_MODEL = "lgbm_market"
# réunions régionales : absentes du programme national de pmu.fr, souvent sans pari en ligne
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
                   rc.c.name.label("racecourse"))
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


def _horse_names(race_id: int) -> dict[int, str]:
    entries = get_context().repo.entries_as_of(race_id)
    return dict(zip(entries["horse_id"], entries["horse"], strict=True)) \
        if not entries.empty else {}


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
    national = national_races(all_races)
    n_regional = len(all_races) - len(national)
    tab_g, tab_value = st.tabs(["3e du modèle, trot G", "Valeur 4e-6e (cote 9-20)"])
    with tab_g:
        _render_trot_g(day, national, n_regional)
    with tab_value:
        _render_value(day, national, n_regional)
    st.caption(
        "Heures de Paris. Le classement tient compte des cotes, il peut changer jusqu'au "
        "départ : consultez la page 10 à 20 minutes avant la course (les données sont "
        "rafraîchies toutes les 15 minutes) et jouez le cheval affiché à ce moment, sans "
        "revenir dessus. Les tests ont été réglés aux rapports de la masse Internet du PMU.")


def _render_trot_g(day: date, national: pd.DataFrame, n_regional: int) -> None:
    st.warning("Règle NON validée : « 3e du modèle, simple gagnant, courses de trot G ». "
               "Sur les réunions du programme national, avec le modèle actuel, elle aurait "
               "donné −11,7 % sur deux ans (407 paris, intervalle de −35 % à +15 %) : aucun "
               "avantage démontré. À suivre de façon fictive, ou pour le plaisir avec de "
               "petites mises fixes.")
    races = rule_races(national)
    if races.empty:
        notice("Aucune course de trot G ce jour-là au programme national : rien à jouer.")
        return

    rows = []
    with st.spinner("Calcul des probabilités…"):
        for r in races.to_dict("records"):
            pred = _predict(int(r["race_id"]), day)
            # règle testée avec le modèle « LightGBM + marché » : sans cotes, pas de choix
            tested = pred.primary == RULE_MODEL and not pred.table.empty
            pick = rule_pick(pred.table) if tested else None
            names = _horse_names(int(r["race_id"]))
            finish = None if pick is None else _finish(r, pick["num_pmu"])
            rows.append({
                "Départ": _paris_time(r["scheduled_start_utc"]),
                "Course": f"R{r['reunion_num']}C{r['course_num']}",
                "Hippodrome": r["racecourse"] or "?",
                "Distance": f"{r['distance_m'] or '?'} m",
                "N° à jouer": "—" if pick is None else int(pick["num_pmu"]),
                "Cheval": "—" if pick is None else names.get(pick["horse_id"], "?"),
                "P(victoire)": "—" if pick is None else f"{100 * pick['p_win']:.0f} %",
                "Cote réf.": "—" if pick is None or pd.isna(pick["mkt_odds"])
                else f"{pick['mkt_odds']:.1f}",
                "Arrivée": "—" if finish is None or pd.isna(finish) else int(finish),
                "Remarque": "" if pick is not None else (
                    "cotes pas encore publiées : revenir plus tard" if not pred.table.empty
                    else pred.message),
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if n_regional:
        st.caption(f"{n_regional} course(s) de réunions régionales exclue(s) : elles ne "
                   "figurent pas au programme national de pmu.fr.")


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


def _render_value(day: date, national: pd.DataFrame, n_regional: int) -> None:
    rule = VALUE_46
    st.warning(
        "Règle NON validée, figée le 21/11/2026 pour être jugée sur les courses à venir : "
        "« 4e à 6e du modèle, cote 9 à 20, 10 à 14 partants, espérance du modèle d'au moins "
        "+10 %, simple gagnant ». Sur deux ans de tests elle a donné +22,6 %, puis +41,8 %, "
        "puis +22,0 % sur les six derniers mois, mais avec une marge d'erreur de −29 % à "
        "+76 % : c'est peut-être de la chance (8 640 stratégies essayées). Environ 800 "
        "paris par an, environ 1 gagnant sur 9. À suivre de façon fictive, ou avec de très "
        "petites mises fixes.")
    if national.empty:
        notice("Aucune course ce jour-là au programme national.")
        return
    now = datetime.now(UTC)
    rows, watch = [], []
    with st.spinner("Calcul des probabilités…"):
        for r in national.to_dict("records"):
            rid = int(r["race_id"])
            pred = _predict(rid, day)
            table = pred.table if pred.primary == RULE_MODEL else pred.table.iloc[0:0]
            entries = get_context().repo.entries_as_of(rid)
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
                "Discipline": DISCIPLINE_NAMES.get(str(r["discipline"]), "?"),
                "Partants": n,
                "État": watch_state(table, picks, started, bool(r["result_final"])),
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
        notice("Aucune course de 10 à 14 partants ce jour-là au programme national.")
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
    notes = ["Courses du programme national de 10 à 14 partants : la règle ne peut retenir "
             "un cheval que dans celles-ci. Le choix dépend des cotes, qui bougent jusqu'au "
             "départ : rafraîchissez la page à l'heure « Consulter vers » et jouez les "
             "chevaux retenus à ce moment-là, sans revenir dessus."]
    if n_regional:
        notes.append(f"{n_regional} course(s) de réunions régionales exclue(s).")
    st.caption(" ".join(notes) + " Une même course peut donner plusieurs chevaux (1 € "
               "chacun dans les tests). L'espérance est calculée sur la cote de référence "
               "× 0,90 : ce n'est pas un gain garanti.")
