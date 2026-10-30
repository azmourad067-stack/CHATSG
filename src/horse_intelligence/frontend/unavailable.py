"""Pages des phases ultérieures : affichent explicitement leur indisponibilité."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from horse_intelligence.frontend.components.cards import notice

_PAGES = {
    "Analyse approfondie": (
        "Phase 2",
        "Aucun modèle n'est entraîné ni validé : aucune probabilité n'est affichée. "
        "Pré-requis : historique suffisant, feature store « as-of », modèles de référence "
        "(uniforme, marché, logit conditionnel) puis LightGBM évalués en walk-forward."),
    "Laboratoire IA": (
        "Phase 3",
        "La découverte de patterns exige une période de découverte et une période de "
        "confirmation indépendantes, avec correction des tests multiples."),
    "Backtesting": (
        "Phase 5",
        "Aucun résultat de backtest n'est affiché tant qu'un modèle n'a pas été évalué sur "
        "des données chronologiquement indépendantes."),
}


def make(title: str) -> Callable[[], None]:
    phase, text = _PAGES[title]

    def render() -> None:
        st.title(title)
        st.caption(f"Disponible en {phase}")
        notice(text)

    render.__name__ = "render_" + title.lower().replace(" ", "_")
    return render
