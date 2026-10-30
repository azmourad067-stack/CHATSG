"""Pages des phases ultérieures : affichent explicitement leur indisponibilité."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from horse_intelligence.frontend.components.cards import notice

_PAGES = {
    "Laboratoire IA": (
        "Phase 3",
        "La découverte de patterns exige une période de découverte et une période de "
        "confirmation indépendantes, avec correction des tests multiples."),
}


def make(title: str) -> Callable[[], None]:
    phase, text = _PAGES[title]

    def render() -> None:
        st.title(title)
        st.caption(f"Disponible en {phase}")
        notice(text)

    render.__name__ = "render_" + title.lower().replace(" ", "_")
    return render
