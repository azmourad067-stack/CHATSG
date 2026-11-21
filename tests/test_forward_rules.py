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
