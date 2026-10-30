"""Tableau de bord."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from horse_intelligence.frontend.components.cards import metric_card, notice
from horse_intelligence.frontend.state import format_dt, get_context


@st.cache_data(ttl=60, show_spinner=False)
def _counts() -> dict[str, int]:
    return get_context().repo.counts()


@st.cache_data(ttl=60, show_spinner=False)
def _last_update() -> datetime | None:
    return get_context().repo.last_update()


@st.cache_data(ttl=60, show_spinner=False)
def _sources() -> pd.DataFrame:
    return get_context().repo.source_statuses()


@st.cache_data(ttl=60, show_spinner=False)
def _runs() -> pd.DataFrame:
    return get_context().repo.recent_runs(10)


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def render() -> None:
    st.title("Tableau de bord")
    ctx = get_context()
    counts = _counts()

    if ctx.settings.using_default_database:
        notice("Base locale SQLite (éphémère sur Streamlit Cloud). Configurez DATABASE_URL "
               "pour une base persistante — voir README.")

    cols = st.columns(4)
    with cols[0]:
        metric_card("Courses en base", _fmt(counts["races"]))
    with cols[1]:
        metric_card("Chevaux étudiés", _fmt(counts["horses"]))
    with cols[2]:
        metric_card("Résultats officiels", _fmt(counts["results"]))
    with cols[3]:
        metric_card("Dernière collecte", format_dt(_last_update()))

    st.write("")
    cols = st.columns(3)
    with cols[0]:
        metric_card("Modèle actif", "Aucun", "Phase 2 : aucun modèle entraîné ni validé")
    with cols[1]:
        metric_card("Performance historique", "Non mesurée",
                    "Aucune métrique sans backtest indépendant")
    with cols[2]:
        metric_card("Apprentissage", "Inactif", "Cycles contrôlés prévus en phase 5")

    st.subheader("État des sources")
    sources = _sources()
    if not sources.empty:
        st.dataframe(
            sources[["source_key", "name", "enabled", "status_audit", "last_status",
                     "last_checked_at", "last_message"]],
            hide_index=True, width="stretch",
            column_config={"source_key": "Clé", "name": "Source", "enabled": "Activée",
                           "status_audit": "Audit", "last_status": "Dernier état",
                           "last_checked_at": "Vérifiée le", "last_message": "Message"},
        )
    st.caption("Tester les sources : page Configuration.")

    st.subheader("Dernières collectes")
    runs = _runs()
    if runs.empty:
        notice("Aucune collecte enregistrée. Lancez : python scripts/refresh_data.py")
    else:
        st.dataframe(runs, hide_index=True, width="stretch")
