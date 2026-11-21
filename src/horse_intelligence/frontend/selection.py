"""Sélection du jour : deux règles NON validées, appliquées telles qu'elles ont été définies.

1. « 3e du modèle, simple gagnant, trot G » (ci-dessous).
2. Profil « 4e à 6e du modèle, cote 20 à 50, trot attelé, 15 partants ou plus, modèle >
   marché, simple gagnant », ajouté à la demande de l'utilisateur le 21/11/2026. Meilleur
   des 3 360 profils sur la 1re moitié (nov. 2024 → nov. 2025 : +50,5 %, 208 paris), il a
   donné −51,1 % sur la 2nde moitié (nov. 2025 → nov. 2026, 319 paris) : −11 % sur deux
   ans (527 paris). Aucun avantage démontré.

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

from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import select

from horse_intelligence.data.storage import database as db
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

# profil « outsiders attelé » (experiment profiles du 21/11/2026)
PROFILE_DISCIPLINE = "trot_attele"
PROFILE_RANKS = (4, 6)            # 4e à 6e du modèle
PROFILE_ODDS = (20.0, 50.0)       # cote de référence > 20 et ≤ 50
PROFILE_MIN_RUNNERS = 15


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


def profile_picks(table: pd.DataFrame) -> pd.DataFrame:
    """Chevaux du profil dans une course (déjà filtrée sur la discipline) : 4e à 6e du
    modèle, cote de référence dans ]20 ; 50], probabilité du modèle > celle du marché."""
    if len(table) < PROFILE_MIN_RUNNERS or table["mkt_odds"].isna().any():
        return table.iloc[0:0]
    t = table.sort_values("p_win", ascending=False, kind="stable").reset_index(drop=True)
    t["model_rank"] = range(1, len(t) + 1)
    lo, hi = PROFILE_RANKS
    return t[t["model_rank"].between(lo, hi) & (t["mkt_odds"] > PROFILE_ODDS[0])
             & (t["mkt_odds"] <= PROFILE_ODDS[1]) & (t["p_win"] > t["mkt_prob"])]


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
    tab_g, tab_profile = st.tabs(["3e du modèle, trot G",
                                  "Outsiders attelé (4e-6e, cote 20-50)"])
    with tab_g:
        _render_trot_g(day, national, n_regional)
    with tab_profile:
        _render_profile(day, national, n_regional)
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


def _render_profile(day: date, national: pd.DataFrame, n_regional: int) -> None:
    st.error("Profil NON validé, et perdant sur la période la plus récente : « 4e à 6e du "
             "modèle, cote 20 à 50, trot attelé, 15 partants ou plus, modèle plus "
             "optimiste que la cote, simple gagnant ». Meilleur des 3 360 profils sur la "
             "1re année testée (+50,5 %, 208 paris), il a donné −51,1 % l'année suivante "
             "(319 paris, seulement 3 % de gagnants) : −11 % sur deux ans. C'est un pari "
             "sur de gros outsiders : de longues séries sans gain (jusqu'à 115 paris de "
             "suite) sont normales. À suivre de façon fictive, ou avec de très petites "
             "mises fixes.")
    races = national[national["discipline"] == PROFILE_DISCIPLINE]
    if races.empty:
        notice("Aucune course de trot attelé ce jour-là au programme national.")
        return
    rows, skipped = [], 0
    with st.spinner("Calcul des probabilités…"):
        for r in races.to_dict("records"):
            pred = _predict(int(r["race_id"]), day)
            if pred.primary != RULE_MODEL or pred.table.empty:
                skipped += 1
                continue
            picks = profile_picks(pred.table)
            if picks.empty:
                continue
            names = _horse_names(int(r["race_id"]))
            for pick in picks.to_dict("records"):
                finish = _finish(r, int(pick["num_pmu"]))
                rows.append({
                    "Départ": _paris_time(r["scheduled_start_utc"]),
                    "Course": f"R{r['reunion_num']}C{r['course_num']}",
                    "Hippodrome": r["racecourse"] or "?",
                    "Partants": len(pred.table),
                    "N° à jouer": int(pick["num_pmu"]),
                    "Cheval": names.get(pick["horse_id"], "?"),
                    "Rang modèle": int(pick["model_rank"]),
                    "P(victoire)": f"{100 * pick['p_win']:.1f} %",
                    "P(marché)": f"{100 * pick['mkt_prob']:.1f} %",
                    "Cote réf.": f"{pick['mkt_odds']:.1f}",
                    "Arrivée": "—" if finish is None or pd.isna(finish) else int(finish),
                })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        notice("Aucun cheval ne correspond au profil ce jour-là (pour l'instant).")
    notes = [f"{len(races)} course(s) de trot attelé examinée(s)."]
    if skipped:
        notes.append(f"{skipped} sans cotes publiées : revenir plus tard.")
    if n_regional:
        notes.append(f"{n_regional} course(s) de réunions régionales exclue(s).")
    st.caption(" ".join(notes) + " Un même course peut donner plusieurs chevaux (1 € chacun "
               "dans le test).")
