"""Point d'entrée Streamlit : `streamlit run app.py`."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from horse_intelligence.frontend import (  # noqa: E402
    ai_laboratory,
    backtesting,
    dashboard,
    horse_profile,
    race_analysis,
    races,
    settings,
)
from horse_intelligence.frontend.components.cards import inject_style  # noqa: E402
from horse_intelligence.monitoring.logging_config import configure_logging  # noqa: E402

st.set_page_config(page_title="Horse Intelligence", page_icon="🐎", layout="wide")
configure_logging()
inject_style()

pages = {
    "Vue d'ensemble": [
        st.Page(dashboard.render, title="Tableau de bord", default=True),
        st.Page(races.render, title="Courses du jour", url_path="courses"),
        st.Page(horse_profile.render, title="Fiche cheval", url_path="cheval"),
    ],
    "Intelligence": [
        st.Page(race_analysis.render, title="Analyse approfondie", url_path="analyse"),
        st.Page(ai_laboratory.render, title="Laboratoire IA", url_path="laboratoire"),
        st.Page(backtesting.render, title="Backtesting", url_path="backtesting"),
    ],
    "Système": [
        st.Page(settings.render, title="Configuration", url_path="configuration"),
    ],
}

nav = st.navigation(pages)
st.sidebar.caption("Outil d'analyse à usage personnel. Estimations sans garantie de gain. "
                   "Jouer comporte des risques : 09 74 75 13 13 (appel non surtaxé).")
nav.run()
