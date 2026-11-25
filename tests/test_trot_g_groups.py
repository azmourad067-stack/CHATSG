"""Trot G sur plusieurs chevaux : sélection des courses, mises, dutching."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import trot_g_groups as tg


def _race(rid: int, winner_rank: int, cond: str = "Course G") -> pd.DataFrame:
    odds = [2.0, 4.0, 8.0, 10.0, 20.0]
    return pd.DataFrame({
        "race_id": rid, "race_date": date(2026, 1, 1), "rank": [1, 2, 3, 4, 5],
        "mkt_odds": odds, "label": "PRIX", "conditions_text": cond,
        "discipline": "trot_attele",
        "profit_gagnant": [o - 1 if r == winner_rank else -1.0
                           for r, o in zip([1, 2, 3, 4, 5], odds, strict=True)]})


def test_group_bets_one_euro_and_dutching() -> None:
    f = pd.concat([_race(1, 3), _race(2, 4), _race(3, 3, cond="Course F")])
    g = tg.trot_g(f)
    assert set(g["race_id"]) == {1, 2}  # course F écartée
    one = tg.group_bets(g, (3, 4), "1 € par cheval").set_index("race_id")
    # course 1 : le 3e gagne à 8 (7 − 1) ; course 2 : le 4e gagne à 10 (9 − 1)
    assert one.loc[1, "stake"] == 2 and one.loc[1, "profit"] == 6.0
    assert one.loc[2, "profit"] == 8.0
    dut = tg.group_bets(g, (3, 4), "dutching").set_index("race_id")
    assert np.isclose(dut.loc[1, "stake"], 2.0)
    # dutching : à peu près le même retour quel que soit le gagnant du groupe
    ret1, ret2 = dut.loc[1, "profit"] + 2, dut.loc[2, "profit"] + 2
    assert np.isclose(ret1, ret2) and np.isclose(ret1, 2 / (1 / 8 + 1 / 10))
    assert tg.group_bets(g, (3,), "1 € par cheval")["profit"].tolist() == [7.0, -1.0]


def test_run_reports_every_strategy() -> None:
    f = pd.concat([_race(i, 1 + i % 5) for i in range(40)])
    rep = tg.run(f)
    assert rep["n_races"] == 40
    assert len(rep["strategies"]) == len(tg.GROUPS) * len(tg.STAKINGS)
    assert [r["rank"] for r in rep["by_rank"]] == [1, 2, 3, 4, 5]
    assert isinstance(rep["success"], bool)
