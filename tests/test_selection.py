"""Page « Sélection du jour » : même filtre et même rang que la règle figée testée."""

from __future__ import annotations

import pandas as pd

from horse_intelligence.evaluation.frozen_rules import FROZEN_RULES
from horse_intelligence.frontend.selection import RULE_GROUP, RULE_RANK, rule_pick, rule_races


def test_rule_matches_frozen_rule() -> None:
    rule = next(r for r in FROZEN_RULES if r.value == RULE_GROUP)
    assert rule.rank == str(RULE_RANK) and rule.bet_type == "SIMPLE_GAGNANT"


def test_only_trot_g_races_are_kept() -> None:
    races = pd.DataFrame({
        "race_id": [1, 2, 3, 4],
        "label": ["PRIX DE VINCENNES", "PRIX X", "PRIX Y", "PRIX Z"],
        "conditions_text": ["Course G - Pour chevaux de 4 ans", "Course D - Attelé",
                            "Course G - Monté", "Handicap - Course G"],
        "discipline": ["trot_attele", "trot_attele", "trot_monte", "plat"],
    })
    assert rule_races(races)["race_id"].tolist() == [1, 3]


def test_pick_is_third_highest_probability() -> None:
    table = pd.DataFrame({"num_pmu": [1, 2, 3, 4, 5], "p_win": [0.10, 0.30, 0.05, 0.25, 0.30]})
    # 0.30 (n°2), 0.30 (n°5), 0.25 (n°4) : le 3e est le n°4
    assert int(rule_pick(table)["num_pmu"]) == 4  # type: ignore[index]
    assert rule_pick(table.head(2)) is None


def test_frame_cache_is_rebuilt_when_new_data_arrives(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Régression : avec une clé « _data_key », Streamlit n'invalidait jamais le cache et les
    cotes collectées après la première ouverture de la page n'étaient jamais vues."""
    from datetime import date

    import streamlit as st

    from horse_intelligence.frontend import race_analysis

    calls: list[date] = []

    def fake_build(engine, day, use_agg):  # type: ignore[no-untyped-def]
        calls.append(day)
        return pd.DataFrame({"n": [len(calls)]})

    class Ctx:
        class repo:
            engine = None

    monkeypatch.setattr(race_analysis, "build_frame", fake_build)
    monkeypatch.setattr(race_analysis, "get_context", lambda: Ctx)
    st.cache_resource.clear()
    day = date(2026, 11, 12)
    race_analysis._frame(day, "v1|18:36", True)
    race_analysis._frame(day, "v1|18:36", True)
    assert len(calls) == 1  # même donnée : cache utilisé
    race_analysis._frame(day, "v1|19:36", True)
    assert len(calls) == 2  # nouvelle collecte : reconstruit
    st.cache_resource.clear()
