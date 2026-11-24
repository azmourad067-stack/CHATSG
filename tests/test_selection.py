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


def test_frozen_rule_split_by_audience(repo) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import update
    from tests.synthetic import populate

    from horse_intelligence.data.storage import database as db
    from horse_intelligence.services.experiment import run_frozen_by_audience
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    with repo.engine.begin() as conn:
        conn.execute(update(db.races).values(audience="NATIONAL"))
        conn.execute(update(db.races).where(db.races.c.course_num == 1)
                     .values(audience="LOCAL"))
    rep = run_frozen_by_audience(repo.engine)
    assert set(rep["n_races"]) == {"nationales", "régionales"}
    assert set(rep["rule"]) == {"toutes", "nationales", "régionales"}
    tot = rep["rule"]["toutes"]["total"]["n_bets"]
    assert tot == (rep["rule"]["nationales"]["total"]["n_bets"]
                   + rep["rule"]["régionales"]["total"]["n_bets"])
    assert set(rep["rule_place"]) == {"toutes", "nationales", "régionales"}


def test_value_tab_uses_the_frozen_rule() -> None:
    from horse_intelligence.evaluation import forward_rules as fr
    from horse_intelligence.frontend import selection as sel

    assert sel.VALUE_46 is fr.VALUE_46 and sel.rule_picks is fr.rule_picks
    assert not hasattr(sel, "profile_picks")  # onglet « outsiders attelé » retiré


def test_watch_list_helpers() -> None:
    from horse_intelligence.frontend.selection import n_starters, watch_state

    entries = pd.DataFrame({"num_pmu": [1, 2, 3], "status": ["PARTANT", "NON_PARTANT", None]})
    assert n_starters(entries) == 2 and n_starters(pd.DataFrame()) == 0
    table = pd.DataFrame({"num_pmu": [4, 7], "mkt_odds": [12.0, 15.0]})
    no_odds = table.assign(mkt_odds=[12.0, float("nan")])
    empty = table.iloc[0:0]
    assert watch_state(table, empty, False, True) == "terminée"
    assert watch_state(no_odds, empty, False, False).startswith("cotes pas encore publiées")
    assert watch_state(empty, empty, True, False) == "partie"
    assert watch_state(table, empty, False, False) == "aucun cheval retenu pour l'instant"
    assert watch_state(table, table.head(1), False, False).endswith(": n°4")


def _value_tab_script() -> None:
    from datetime import date

    import pandas as pd

    from horse_intelligence.frontend import selection as sel

    races = pd.DataFrame({
        "race_id": [1, 2, 3], "reunion_num": [1, 1, 2], "course_num": [1, 2, 1],
        "label": ["A", "B", "C"], "conditions_text": ["", "", ""],
        "discipline": ["trot_attele", "plat", "plat"], "distance_m": [2700, 1600, 1600],
        "scheduled_start_utc": pd.to_datetime(["2099-01-01 12:00", "2099-01-01 13:00",
                                               "2099-01-01 14:00"]),
        "result_final": [False, False, False],
        "audience": ["NATIONAL", "REGIONAL", "LOCAL"],
        "racecourse": ["VINCENNES", "PAU", "PAU"]})
    sel._render_value(races, sel._day_data(date(2099, 1, 1), races))


def test_value_tab_lists_races_to_watch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from horse_intelligence.frontend import selection as sel

    p = [0.25, 0.18, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.04]
    odds = [3.0, 5.0, 8.0, 12.0, 20.0, 9.0, 15.0, 18.0, 30.0, 40.0]
    table = pd.DataFrame({"num_pmu": range(1, 11), "horse_id": range(101, 111), "p_win": p,
                          "mkt_odds": odds, "mkt_prob": p})

    def fake_day_predictions(day, race_ids):  # type: ignore[no-untyped-def]
        assert sorted(race_ids) == [1, 2, 3]  # un seul calcul pour toutes les courses
        return {
            # 10 partants, cotes publiées : le n°5 est retenu
            1: SimpleNamespace(primary=sel.RULE_MODEL, table=table, message=""),
            # cotes pas encore publiées
            2: SimpleNamespace(primary="lgbm_form", table=table.iloc[0:0], message=""),
            3: SimpleNamespace(primary=sel.RULE_MODEL, table=table.head(8), message="")}

    one = pd.DataFrame({"num_pmu": range(1, 13), "horse_id": range(101, 113),
                        "horse": [f"CHEVAL {i}" for i in range(1, 13)],
                        "status": ["PARTANT"] * 12})
    entries = pd.concat([one.head(10).assign(race_id=1), one.assign(race_id=2),
                         one.head(8).assign(race_id=3)])
    repo = SimpleNamespace(engine=None, entries_for_races=lambda ids: entries,
                           results_for=lambda rid: pd.DataFrame())
    monkeypatch.setattr(sel, "day_predictions", fake_day_predictions)
    monkeypatch.setattr(sel, "races_with_internet_odds", lambda engine, ids: {1, 2, 3})
    monkeypatch.setattr(sel, "get_context", lambda: SimpleNamespace(repo=repo))
    at = AppTest.from_function(_value_tab_script, default_timeout=30)
    at.run()
    assert not at.exception
    picks, watch = at.dataframe[0].value, at.dataframe[1].value
    assert picks["N° à jouer"].tolist() == [5] and picks["Cheval"].tolist() == ["CHEVAL 5"]
    # course 3 (8 partants) absente ; course 2 (12 partants, sans cotes) à surveiller
    assert watch["Course"].tolist() == ["R1C1", "R1C2"]
    assert watch["État"].iloc[1].startswith("cotes pas encore publiées")
    assert watch["Réunion"].tolist() == ["nationale", "régionale"]  # régionales incluses
    assert watch["Consulter vers"].iloc[0] != "—"
    assert any("Prochaine course à surveiller : R1C1" in i.value for i in at.info)
    assert any(i.value.startswith("Seul le SMS fait foi") for i in at.info)


def _small_tab_script() -> None:
    from datetime import date

    import pandas as pd

    from horse_intelligence.frontend import selection as sel

    races = pd.DataFrame({
        "race_id": [1, 2, 3, 4], "reunion_num": [1, 5, 1, 6], "course_num": [1, 1, 2, 1],
        "label": ["A", "B", "C", "D"], "conditions_text": [""] * 4,
        "discipline": ["plat", "trot_attele", "plat", "plat"],
        "distance_m": [1600, 2700, 1600, 1600],
        "scheduled_start_utc": pd.to_datetime(["2099-01-01 12:00", "2099-01-01 13:00",
                                               "2099-01-01 14:00", "2099-01-01 15:00"]),
        "result_final": [False] * 4, "audience": ["NATIONAL", "REGIONAL", "NATIONAL", "LOCAL"],
        "prize_eur": [20000.0, 30000.0, 8000.0, 5000.0],
        "racecourse": ["VINCENNES", "PAU", "CAEN", "SALON"]})
    sel._render_small(races, sel._day_data(date(2099, 1, 1), races))


def test_small_races_tab_applies_the_frozen_rule(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from horse_intelligence.frontend import selection as sel

    p = [0.25, 0.18, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.04]
    odds = [3.0, 5.0, 8.0, 12.0, 20.0, 9.0, 15.0, 18.0, 30.0, 40.0]
    table = pd.DataFrame({"num_pmu": range(1, 11), "horse_id": range(101, 111), "p_win": p,
                          "mkt_odds": odds, "mkt_prob": p})
    preds = {rid: SimpleNamespace(primary=sel.RULE_MODEL, table=table, message="")
             for rid in (1, 2, 3)}
    preds[4] = SimpleNamespace(primary="lgbm_form", table=table.iloc[0:0], message="")
    entries = pd.concat([pd.DataFrame({"race_id": rid, "num_pmu": range(1, 11),
                                       "horse_id": range(101, 111),
                                       "horse": [f"CHEVAL {i}" for i in range(1, 11)],
                                       "status": "PARTANT"}) for rid in (1, 2, 3, 4)])
    repo = SimpleNamespace(engine=None, entries_for_races=lambda ids: entries,
                           results_for=lambda rid: pd.DataFrame())
    monkeypatch.setattr(sel, "day_predictions", lambda day, ids: preds)
    # course 3 (8 000 €) sans cote Internet : pas de pari en ligne
    monkeypatch.setattr(sel, "races_with_internet_odds", lambda engine, ids: {1, 2, 4})
    monkeypatch.setattr(sel, "first_time_d4", lambda engine, ids, day: set())
    monkeypatch.setattr(sel, "get_context", lambda: SimpleNamespace(repo=repo))
    at = AppTest.from_function(_small_tab_script, default_timeout=30)
    at.run()
    assert not at.exception
    picks, watch = at.dataframe[0].value, at.dataframe[1].value
    # seul le n°5 (cote 20, espérance +62 %) de la course régionale est retenu
    assert picks["Course"].tolist() == ["R5C1"] and picks["N° à jouer"].tolist() == [5]
    # la course nationale à 20 000 € n'est pas une petite course
    assert watch["Course"].tolist() == ["R5C1", "R1C2", "R6C1"]
    assert watch["État"].iloc[1].startswith("pas de pari en ligne")
    assert watch["État"].iloc[2].startswith("cotes pas encore publiées")
    assert watch["Décider entre"].iloc[0] == "13:32 et 13:47"
    assert at.warning[0].value.startswith("Règle NON validée, figée le 24/11/2026")


def _d4_tab_script() -> None:
    from datetime import date

    import pandas as pd

    from horse_intelligence.frontend import selection as sel

    races = pd.DataFrame({
        "race_id": [1, 2, 3], "reunion_num": [1, 3, 4], "course_num": [1, 2, 3],
        "label": ["A", "B", "C"],
        "conditions_text": ["Départ à l'autostart", "AUTOSTART", "Départ volté"],
        "discipline": ["trot_attele"] * 3, "distance_m": [2700, 2100, 2850],
        "scheduled_start_utc": pd.to_datetime(["2099-01-01 12:00", "2099-01-01 13:00",
                                               "2099-01-01 14:00"]),
        "result_final": [False] * 3, "audience": ["NATIONAL", "LOCAL", "NATIONAL"],
        "racecourse": ["VINCENNES", "CORDEMAIS", "CAEN"]})
    sel._render_d4(races, sel._day_data(date(2099, 1, 1), races))


def test_d4_tab_applies_the_frozen_autostart_rule(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from horse_intelligence.frontend import selection as sel

    table = pd.DataFrame({"num_pmu": [1, 2, 3, 4, 5, 6],
                          "horse_id": [11, 12, 13, 14, 15, 16],
                          "p_win": [0.30, 0.25, 0.15, 0.12, 0.10, 0.08],
                          "mkt_odds": [3.0, 4.0, 7.0, 8.0, 6.0, 12.0],
                          "mkt_prob": [0.3, 0.25, 0.15, 0.12, 0.1, 0.08]})
    preds = {1: SimpleNamespace(primary=sel.RULE_MODEL, table=table, message=""),
             2: SimpleNamespace(primary=None, table=pd.DataFrame(), message=""),
             3: SimpleNamespace(primary=sel.RULE_MODEL, table=table, message="")}
    names = ["UN", "DEUX", "TROIS", "QUATRE", "CINQ", "SIX"]
    entries = pd.concat([
        pd.DataFrame({"race_id": 1, "num_pmu": range(1, 7), "horse_id": range(11, 17),
                      "horse": names, "status": "PARTANT",
                      # n°2 part en 8 (corde publiée), n°6 en 4, n°5 sans corde : place 5
                      "draw": [None, 8, None, None, None, 4]}),
        pd.DataFrame({"race_id": [2], "num_pmu": [4], "horse_id": [24], "horse": ["AUTRE"],
                      "status": ["PARTANT"], "draw": [None]}),
        pd.DataFrame({"race_id": [3], "num_pmu": [5], "horse_id": [35], "horse": ["VOLTE"],
                      "status": ["PARTANT"], "draw": [None]})])
    repo = SimpleNamespace(engine=None, entries_for_races=lambda ids: entries,
                           results_for=lambda rid: pd.DataFrame())
    monkeypatch.setattr(sel, "day_predictions", lambda day, ids: preds)
    monkeypatch.setattr(sel, "races_with_internet_odds", lambda engine, ids: {1, 3})
    monkeypatch.setattr(sel, "first_time_d4", lambda engine, ids, day: {
        (1, 2), (1, 5), (1, 6), (2, 4), (3, 5)})
    monkeypatch.setattr(sel, "get_context", lambda: SimpleNamespace(repo=repo))
    at = AppTest.from_function(_d4_tab_script, default_timeout=30)
    at.run()
    assert not at.exception
    df = at.dataframe[0].value
    # n°2 (place 8) et la course au départ volté sont écartés
    assert df["Cheval"].tolist() == ["CINQ", "SIX", "AUTRE"]
    assert df["Place départ"].tolist() == ["5", "4", "4"]
    assert df["Rang modèle"].tolist() == ["5", "6", "—"]
    assert df["État"].iloc[0].startswith("RETENU")
    assert df["État"].iloc[1].startswith("cote 12.0 au-dessus de 9")
    assert df["État"].iloc[2].startswith("cotes pas encore publiées")
    assert at.warning[0].value.startswith("Règle NON validée")


def _trot_g_tab_script() -> None:
    from datetime import date

    import pandas as pd

    from horse_intelligence.frontend import selection as sel

    races = pd.DataFrame({
        "race_id": [1, 2, 3], "reunion_num": [1, 9, 10], "course_num": [1, 2, 3],
        "label": ["PRIX A", "PRIX B", "PRIX C"],
        "conditions_text": ["Course G - Attelé"] * 3,
        "discipline": ["trot_attele"] * 3, "distance_m": [2700, 2100, 2100],
        "scheduled_start_utc": pd.to_datetime(["2099-01-01 12:00", "2099-01-01 13:00",
                                               "2099-01-01 14:00"]),
        "result_final": [False] * 3, "audience": ["NATIONAL", "REGIONAL", "LOCAL"],
        "racecourse": ["VINCENNES", "SALON", "CORDEMAIS"]})
    sel._render_trot_g(races, sel._day_data(date(2099, 1, 1), races))


def test_trot_g_tab_includes_online_regional_races(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from horse_intelligence.frontend import selection as sel

    table = pd.DataFrame({"num_pmu": [1, 2, 3, 4], "horse_id": [11, 12, 13, 14],
                          "p_win": [0.4, 0.3, 0.2, 0.1], "mkt_odds": [2.0, 4.0, 6.0, 9.0],
                          "mkt_prob": [0.4, 0.3, 0.2, 0.1]})
    preds = {rid: SimpleNamespace(primary=sel.RULE_MODEL, table=table, message="")
             for rid in (1, 2, 3)}
    entries = pd.concat([pd.DataFrame({"race_id": rid, "num_pmu": [1, 2, 3, 4],
                                       "horse_id": [11, 12, 13, 14],
                                       "horse": ["UN", "DEUX", "TROIS", "QUATRE"],
                                       "status": "PARTANT"}) for rid in (1, 2, 3)])
    repo = SimpleNamespace(engine=None, entries_for_races=lambda ids: entries,
                           results_for=lambda rid: pd.DataFrame())
    monkeypatch.setattr(sel, "day_predictions", lambda day, ids: preds)
    # course locale n°3 : pas de cote Internet, donc pas de pari en ligne
    monkeypatch.setattr(sel, "races_with_internet_odds", lambda engine, ids: {1, 2})
    monkeypatch.setattr(sel, "first_time_d4", lambda engine, ids, day: set())
    monkeypatch.setattr(sel, "get_context", lambda: SimpleNamespace(repo=repo))
    at = AppTest.from_function(_trot_g_tab_script, default_timeout=30)
    at.run()
    assert not at.exception
    df = at.dataframe[0].value
    assert df["Réunion"].tolist() == ["nationale", "régionale", "locale"]
    assert df["N° à jouer"].tolist() == ["3", "3", "—"]  # 3e du modèle
    assert df["Remarque"].iloc[2].startswith("pas de pari en ligne")
