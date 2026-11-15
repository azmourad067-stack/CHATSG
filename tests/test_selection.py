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


def test_regional_reunions_are_excluded() -> None:
    from horse_intelligence.frontend.selection import national_races

    races = pd.DataFrame({"race_id": [1, 2, 3, 4],
                          "audience": ["NATIONAL", "LOCAL", "REGIONAL", None]})
    # audience inconnue (courses collectées avant le 15/11/2026) : conservée
    assert national_races(races)["race_id"].tolist() == [1, 4]


def test_audience_is_collected_and_old_databases_are_upgraded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import create_engine, inspect, text
    from tests.conftest import RACE_DAY, programme_payload

    from horse_intelligence.data.processors.normalization import race_from_pmu
    from horse_intelligence.data.schemas.pmu import PmuProgramme
    from horse_intelligence.data.storage.database import init_db

    reunion = PmuProgramme.model_validate(programme_payload()).programme.reunions[0]
    assert race_from_pmu(RACE_DAY, reunion, reunion.courses[0]).audience == "NATIONAL"
    # base créée avant l'ajout de la colonne : init_db l'ajoute sans perdre de données
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE races (race_id INTEGER PRIMARY KEY, "
                          "race_date DATE NOT NULL)"))
        conn.execute(text("INSERT INTO races VALUES (1, '2026-11-01')"))
    init_db(engine)
    assert "audience" in {c["name"] for c in inspect(engine).get_columns("races")}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM races")).scalar_one() == 1
