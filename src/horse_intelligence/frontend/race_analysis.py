"""Analyse approfondie d'une course : probabilités, places, comparaison au marché."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from horse_intelligence.frontend.components.cards import (
    ELECTRIC,
    EMERALD,
    metric_card,
    notice,
    status_badge,
)
from horse_intelligence.frontend.state import format_dt, get_context
from horse_intelligence.prediction.race_predictor import RacePrediction, predict_race

MODEL_LABELS = {"lgbm_market": "LightGBM + marché", "lgbm_form": "LightGBM (forme seule)",
                "market_calibrated": "Marché calibré"}


@st.cache_data(ttl=300, show_spinner=False)
def _predict(race_id: int, _cache_key: str) -> RacePrediction:
    return predict_race(get_context().repo.engine, race_id)


def _pct(p: float | None) -> str:
    return "—" if p is None or pd.isna(p) else f"{100 * p:.0f} %"


def _render_prediction(pred: RacePrediction, horses: dict[int, str]) -> None:
    st.markdown(status_badge(pred.status), unsafe_allow_html=True)
    if pred.table.empty:
        notice(pred.message)
        return
    meta = pred.models[pred.primary or ""]
    if meta["status"] == "active":
        st.success(pred.message)
    else:
        st.warning(pred.message)

    unc_text = {"exploitable": "exploitable", "incertaine": "incertaine",
                "hors distribution": "hors distribution", "non évaluée": "non évaluée"}
    if pred.uncertainty in ("incertaine", "hors distribution"):
        st.warning(f"Course {unc_text[pred.uncertainty]} : " + " ; ".join(
            pred.uncertainty_reasons) + ". Aucun signal exploitable n'est produit.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Modèle principal", MODEL_LABELS.get(pred.primary or "", "?"),
                    f"version {meta['version']} · {meta['status']}")
    with c2:
        metric_card("Places payées", str(pred.n_places) if pred.n_places else "aucune",
                    "simple placé, selon le nombre de partants")
    with c3:
        metric_card("Cotes de référence", "disponibles" if pred.has_market else "absentes",
                    "marché utilisé comme référence")
    with c4:
        metric_card("Incertitude", unc_text.get(pred.uncertainty, pred.uncertainty),
                    "ensemble bootstrap et profil de course")

    t = pred.table.copy()
    t["cheval"] = t["horse_id"].map(horses)
    view = pd.DataFrame({
        "Rang": t["rank"].astype(int), "N°": t["num_pmu"], "Cheval": t["cheval"],
        "P(victoire)": t["p_win"].map(_pct),
        "Intervalle": [f"{_pct(a)} – {_pct(b)}" for a, b in
                       zip(t["p_low"], t["p_high"], strict=True)],
        "P(place)": t["p_place"].map(_pct),
        "P(marché)": t["mkt_prob"].map(_pct),
        "Cote réf.": t["mkt_odds"].map(lambda x: "—" if pd.isna(x) else f"{x:.1f}"),
        "Désaccord": t["disagreement"].map(_pct),
        "Musique": t["form_string"],
    })
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption("Probabilités arrondies au point : la précision réelle est bien moindre. "
               "Intervalle = 10e-90e centiles d'un ensemble de modèles appris sur des "
               "rééchantillonnages de courses (incertitude d'estimation). « Désaccord » = écart "
               "maximal entre les estimations des modèles et du marché.")

    st.subheader("Forces, faiblesses et signaux faibles")
    for r in t.to_dict("records"):
        with st.expander(f"N° {r['num_pmu']} — {r['cheval'] or '?'} · {_pct(r['p_win'])}"):
            st.markdown(f"**Forces** : {r['strengths'] or '—'}")
            st.markdown(f"**Faiblesses** : {r['weaknesses'] or '—'}")
            st.markdown(f"**Signaux faibles validés** : {r['weak_signals'] or 'aucun'}")
    st.caption("Forces et faiblesses : variables qui font monter ou baisser l'estimation par "
               "rapport aux autres partants (contributions SHAP du modèle) ; elles décrivent le "
               "modèle, pas une cause. Signaux faibles : uniquement les patterns validés hors "
               "échantillon par le Laboratoire IA.")

    fig = go.Figure()
    fig.add_bar(name="Modèle", x=t["num_pmu"].astype(str), y=t["p_win"], marker_color=EMERALD)
    if pred.has_market:
        fig.add_bar(name="Marché", x=t["num_pmu"].astype(str), y=t["mkt_prob"],
                    marker_color=ELECTRIC)
    fig.update_layout(barmode="group", height=300, template="plotly_dark",
                      margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_tickformat=".0%",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="N°")
    st.plotly_chart(fig, width="stretch")

    if pred.has_market:
        st.subheader("Comparaison avec les cotes")
        flagged = t[t["value_signal"]]
        if meta["status"] != "active":
            notice("Signal de valeur désactivé : le modèle n'est pas validé. Un écart avec le "
                   "marché n'a pas de valeur démontrée.")
        elif flagged.empty:
            notice("Aucun écart supérieur à l'incertitude des estimations.")
        else:
            st.dataframe(pd.DataFrame({
                "N°": flagged["num_pmu"], "Cheval": flagged["cheval"],
                "Écart": flagged["edge"].map(_pct),
                "Espérance théorique (1 €)": flagged["ev_theoretical"].map(lambda x: f"{x:+.2f} €"),
            }), hide_index=True, width="stretch")
        st.caption("Espérance calculée sur la cote de référence ; au pari mutuel, le rapport "
                   "final diffère. Une espérance estimée positive n'est jamais une garantie.")


def render() -> None:
    st.title("Analyse approfondie")
    repo = get_context().repo
    dates = repo.available_dates()
    if not dates:
        notice("Aucune course en base.")
        return
    day = st.date_input("Date", value=dates[0], min_value=dates[-1], max_value=dates[0],
                        format="DD/MM/YYYY", key="analysis_date")
    races = repo.races_on(day if isinstance(day, date) else dates[0])
    if races.empty:
        notice("Aucune course collectée pour cette date.")
        return
    labels = {int(r["race_id"]): f"R{r['reunion_num']}C{r['course_num']} — "
              f"{r['racecourse'] or ''} — {r['label'] or ''}" for r in races.to_dict("records")}
    race_id = st.selectbox("Course", list(labels), format_func=labels.__getitem__)
    row = races[races["race_id"] == race_id].iloc[0]
    st.caption(f"{row['discipline'] or '?'} · {row['distance_m'] or '?'} m · "
               f"départ {format_dt(row['scheduled_start_utc'])}")
    entries = repo.entries_as_of(int(race_id))
    horses = dict(zip(entries["horse_id"], entries["horse"], strict=True)) \
        if not entries.empty else {}
    with st.spinner("Calcul des probabilités…"):
        pred = _predict(int(race_id), str(repo.last_update()))
    _render_prediction(pred, horses)
