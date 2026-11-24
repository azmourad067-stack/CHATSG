"""Règle figée « Valeur 4e-6e » : mêmes choix que la grande recherche, jugement à venir."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import forward_rules as fr
from horse_intelligence.evaluation import grand_search as gs
from horse_intelligence.evaluation.profile_search import prepare


def _table(odds: list[float], p: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"num_pmu": range(1, len(p) + 1), "p_win": p, "mkt_odds": odds})


def test_rule_picks_rank_odds_ev_and_runners() -> None:
    p = [0.25, 0.18, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.04]
    odds = [3.0, 5.0, 8.0, 12.0, 20.0, 9.0, 15.0, 18.0, 30.0, 40.0]
    picks = fr.rule_picks(_table(odds, p))
    # 4e : 0,10 × 12 × 0,9 − 1 = +8 % (< 10 %) ; 5e : 0,09 × 20 × 0,9 − 1 = +62 % (cote 20
    # incluse) ; 6e : cote 9 exclue
    assert picks["num_pmu"].tolist() == [5] and picks["model_rank"].tolist() == [5]
    assert fr.rule_picks(_table(odds[:9], p[:9])).empty          # 9 partants
    assert fr.rule_picks(_table([*odds[:-1], np.nan], p)).empty  # cote manquante


def test_frozen_rule_selects_like_the_grand_search_strategy() -> None:
    from tests.test_grand_search import _data

    pred, pays = _data(40)
    f = prepare(pred, pays, "model")
    key = ((4, 6), (9.0, 20.0), None, (10, 14), "espérance ≥ +10 %", "gagnant")
    expected = gs.simple_bets(f, "model", gs.Strategy("simple", "", key))
    got = 0
    for _, g in f[f["profit_gagnant"].notna()].groupby("race_id"):
        got += len(fr.rule_picks(g.rename(columns={"model": "p_win"})))
    assert got == expected["stake"].sum() > 0


def test_forward_bets_use_only_predictions_after_freeze(repo) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import insert, select
    from tests.synthetic import populate

    from horse_intelligence.data.storage import database as db

    populate(repo, days=3)
    with repo.engine.begin() as conn:
        races = conn.execute(select(db.races.c.race_id, db.races.c.race_date,
                                    db.races.c.scheduled_start_utc)).all()
        model_id = conn.execute(insert(db.model_versions).values(
            family="lgbm_market", version="t", status="active",
            created_at=datetime.now(UTC))).inserted_primary_key[0]
    assert races
    rule = fr.ValueRule("test", races[0].race_date - timedelta(days=1), (1, 12),
                        (0.0, 999.0), (1, 99), -1.0)
    with repo.engine.begin() as conn:
        for rid, _, start in races:
            pid = conn.execute(insert(db.predictions).values(
                race_id=rid, model_id=model_id, decision_time=start - timedelta(minutes=15),
                created_at=start, data_status="ok", input_snapshot_hash="h"
            )).inserted_primary_key[0]
            entries = conn.execute(select(db.race_entries.c.num_pmu)
                                   .where(db.race_entries.c.race_id == rid)).scalars().all()
            conn.execute(insert(db.prediction_entries), [
                {"prediction_id": pid, "num_pmu": n, "p_win": 1 / len(entries), "rank": 1,
                 "mkt_odds": 5.0} for n in entries])
    bets = fr.forward_bets(repo.engine, rule, national_only=False)
    assert not bets.empty and (bets["race_date"] >= rule.first_day).all()
    v = fr.verdict(bets)
    assert v["bets"] == len(bets) and v["status"].startswith("intermédiaire")


def test_d4_rule_helpers() -> None:
    assert fr.is_autostart("Départ à l'AUTOSTART") and not fr.is_autostart(None)
    assert not fr.is_autostart("Départ volté")
    assert fr.start_place(None, 5) == 5.0 and fr.start_place(3, 5) == 3.0
    assert fr.start_place(float("nan"), "6") == 6.0
    ok = fr.d4_rule_match
    assert ok(5, 6.0, True, True)
    assert ok(4, 9.0, True, True) and ok(6, 1.5, True, True)
    assert not ok(7, 6.0, True, True) and not ok(3, 6.0, True, True)
    assert not ok(5, 9.1, True, True) and not ok(5, 1.0, True, True)
    assert not ok(5, None, True, True) and not ok(5, float("nan"), True, True)
    assert not ok(5, 6.0, False, True) and not ok(5, 6.0, True, False)


def test_d4_rule_selects_like_the_d4_draw_strategy() -> None:
    from horse_intelligence.evaluation import d4_draw

    rng = np.random.default_rng(3)
    n = 6000
    f = pd.DataFrame({
        "num_pmu": rng.integers(1, 16, n),
        "draw": np.where(rng.random(n) < 0.5, rng.integers(1, 16, n), np.nan),
        "mkt_odds": np.round(rng.uniform(1.0, 30.0, n), 1),
        "autostart": rng.integers(0, 2, n).astype(float),
        "shoes_d4_first": rng.integers(0, 2, n).astype(float),
        "profit_gagnant": rng.choice([-1.0, 4.0], n)})
    pos = d4_draw.position(f).to_numpy(float)
    d4 = (f["shoes_d4_first"] == 1).to_numpy()
    rule = fr.D4_AUTOSTART
    expected = d4_draw._mask(f, pos, d4, (rule.places, 1.0, rule.odds))
    got = [fr.d4_rule_match(fr.start_place(r.draw, r.num_pmu), r.mkt_odds,
                            r.autostart == 1, r.shoes_d4_first == 1)
           for r in f.itertuples()]
    assert expected.sum() > 50
    assert got == expected.tolist()


def test_forward_d4_bets_first_time_d4_in_places_4_to_6(repo) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import insert, select, update
    from tests.synthetic import populate

    from horse_intelligence.data.storage import database as db
    from horse_intelligence.prediction.signals import D4

    populate(repo, days=3)
    r, e = db.races, db.race_entries
    with repo.engine.begin() as conn:
        days = sorted(set(conn.execute(select(r.c.race_date)).scalars()))
        conn.execute(update(r).values(conditions_text="Départ à l'autostart"))
        day2 = [rid for (rid,) in conn.execute(select(r.c.race_id)
                                               .where(r.c.race_date == days[1]))]
        conn.execute(update(e).where(e.c.race_id.in_(day2)).values(shoes=D4))
        races = conn.execute(select(r.c.race_id, r.c.scheduled_start_utc)
                             .where(r.c.race_date >= days[1])).all()
        model_id = conn.execute(insert(db.model_versions).values(
            family="lgbm_market", version="t", status="active",
            created_at=datetime.now(UTC))).inserted_primary_key[0]
        for rid, start in races:
            pid = conn.execute(insert(db.predictions).values(
                race_id=rid, model_id=model_id, decision_time=start - timedelta(minutes=15),
                created_at=start, data_status="ok", input_snapshot_hash="h"
            )).inserted_primary_key[0]
            nums = conn.execute(select(e.c.num_pmu).where(e.c.race_id == rid)).scalars().all()
            conn.execute(insert(db.prediction_entries), [
                {"prediction_id": pid, "num_pmu": n, "p_win": 1 / len(nums), "rank": 1,
                 "mkt_odds": 5.0 if n != 6 else 12.0} for n in set(nums)])
    rule = fr.D4StartRule("test", days[0], (4, 6), (1.0, 9.0))
    bets = fr.forward_d4(repo.engine, rule)
    assert not bets.empty
    # seuls les déferrés 4 de la 2e journée (ferrés à leur course précédente) sont joués ;
    # le n°6 (cote 12) est écarté par la cote, la 3e journée (déjà déferrés) aussi
    assert set(bets["race_date"]) == {days[1]}
    assert set(bets["num_pmu"]) <= {4, 5}
    assert (bets["profit"] >= -1).all() and (bets["mkt_odds"] <= 9).all()


def test_early_rule_decides_on_the_prediction_45_to_60_min_before(repo) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import insert, select
    from tests.synthetic import populate

    from horse_intelligence.data.storage import database as db

    populate(repo, days=2)
    with repo.engine.begin() as conn:
        rid, day, start = conn.execute(select(db.races.c.race_id, db.races.c.race_date,
                                              db.races.c.scheduled_start_utc)
                                       .order_by(db.races.c.race_id).limit(1)).one()
        model_id = conn.execute(insert(db.model_versions).values(
            family="lgbm_market", version="t", status="active",
            created_at=datetime.now(UTC))).inserted_primary_key[0]
        nums = conn.execute(select(db.race_entries.c.num_pmu).distinct()
                            .where(db.race_entries.c.race_id == rid)).scalars().all()
        # 55 min avant : cote 12 ; 15 min avant : cote 6 (le cheval a été joué)
        for lead, odds in ((70, 30.0), (55, 12.0), (15, 6.0)):
            pid = conn.execute(insert(db.predictions).values(
                race_id=rid, model_id=model_id, decision_time=start - timedelta(minutes=lead),
                created_at=start, data_status="ok", input_snapshot_hash="h"
            )).inserted_primary_key[0]
            conn.execute(insert(db.prediction_entries), [
                {"prediction_id": pid, "num_pmu": n, "p_win": 1 / len(nums), "rank": 1,
                 "mkt_odds": odds} for n in nums])
    early = fr._last_predictions(repo.engine, day, (45, 60))
    assert set(early["mkt_odds"]) == {12.0}
    assert set(fr._last_predictions(repo.engine, day)["mkt_odds"]) == {6.0}
    assert fr._last_predictions(repo.engine, day, (75, 90)).empty


def test_small_value_rule_matches_the_walk_forward_test() -> None:
    from horse_intelligence.evaluation import small_meetings as sm

    r = fr.SMALL_VALUE
    assert r.small_races and r.lead_minutes == (13, 28) and r.min_ev == sm.MIN_EV
    assert r.odds == sm.ODDS and r.payout_factor == gs.PAYOUT_FACTOR
    assert fr.SMALL_PRIZE == sm.SMALL_PRIZE and fr.SMALL_AUDIENCES == sm.SMALL_AUDIENCES
    assert fr.is_small_race("REGIONAL", 50000) and fr.is_small_race("NATIONAL", 9000)
    assert not fr.is_small_race("NATIONAL", 20000) and not fr.is_small_race(None, None)
    # mêmes chevaux que la règle du test (tous rangs, tous partants)
    t = _table([3.0, 12.0, 25.0, 1.8, 15.0], [0.30, 0.12, 0.06, 0.40, 0.04])
    picks = fr.rule_picks(t, r)
    expect = sm.value_mask(t["p_win"].to_numpy(), t["mkt_odds"].to_numpy())
    assert sorted(picks["num_pmu"]) == sorted(t.loc[expect, "num_pmu"])


def test_small_value_forward_bets_keep_only_small_races(repo) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import insert, select, update
    from tests.synthetic import populate

    from horse_intelligence.data.storage import database as db

    populate(repo, days=3)
    r = db.races
    with repo.engine.begin() as conn:
        rows = conn.execute(select(r.c.race_id, r.c.race_date, r.c.scheduled_start_utc,
                                   r.c.course_num)).all()
        conn.execute(update(r).where(r.c.course_num == 1).values(audience="REGIONAL"))
        conn.execute(update(r).where(r.c.course_num == 2).values(prize_eur=8000.0))
        model_id = conn.execute(insert(db.model_versions).values(
            family="lgbm_market", version="t", status="active",
            created_at=datetime.now(UTC))).inserted_primary_key[0]
        for rid, _, start, _ in rows:
            nums = conn.execute(select(db.race_entries.c.num_pmu).distinct().where(
                db.race_entries.c.race_id == rid)).scalars().all()
            pid = conn.execute(insert(db.predictions).values(
                race_id=rid, model_id=model_id, decision_time=start - timedelta(minutes=20),
                created_at=start, data_status="ok", input_snapshot_hash="h"
            )).inserted_primary_key[0]
            conn.execute(insert(db.prediction_entries), [
                {"prediction_id": pid, "num_pmu": n, "p_win": 0.15, "rank": 1,
                 "mkt_odds": 10.0} for n in nums])
    rule = fr.ValueRule("test", rows[0].race_date - timedelta(days=1), (1, 99), (2.0, 20.0),
                        (0, 99), 0.10, lead_minutes=(13, 28), small_races=True)
    bets = fr.forward_bets(repo.engine, rule)
    courses = {c for rid, _, _, c in rows if rid in set(bets["race_id"])}
    assert not bets.empty and courses <= {1, 2}
    # sans le filtre : toutes les courses
    assert set(fr.forward_bets(repo.engine, fr.ValueRule(
        "t2", rule.frozen_on, (1, 99), (2.0, 20.0), (0, 99), 0.10, lead_minutes=(13, 28)
    ))["race_id"]) > set(bets["race_id"])


def test_small_value_early_differs_only_by_decision_time() -> None:
    a, b = fr.SMALL_VALUE, fr.SMALL_VALUE_EARLY
    assert b.lead_minutes == (45, 60) and b.frozen_on == a.frozen_on
    for f in ("ranks", "odds", "runners", "min_ev", "payout_factor", "bet_type",
              "small_races"):
        assert getattr(a, f) == getattr(b, f)
    from horse_intelligence import alerts

    assert all(ar.rule is not b for ar in alerts.ALERT_RULES)  # pas de SMS
