"""Règles figées : figées, évaluées sans réglage, verdict selon le critère annoncé."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from horse_intelligence.evaluation.frozen_rules import FROZEN_RULES, evaluate_frozen_rules


def test_rules_are_frozen() -> None:
    assert [r.name for r in FROZEN_RULES] == [
        "4e du modèle, placé, quinté", "4e du modèle, gagnant, 1601-2100 m",
        "2e du modèle, placé, réunions R5+", "2e du modèle, gagnant, trot B",
        "3e du modèle, gagnant, trot G"]


def test_r5_rule_is_settled_on_official_payouts() -> None:
    rows, pays = [], []
    for r in range(120):
        day = date(2025, 1, 1) + timedelta(days=r)
        for num, p in ((1, 0.5), (2, 0.3), (3, 0.2)):
            rows.append({"race_id": r, "race_date": day, "num_pmu": num, "lgbm_market": p,
                         "n_runners": 3, "distance_m": 3000, "discipline": "plat",
                         "reunion_num": 6, "label": "PRIX", "conditions_text": ""})
        pays += [{"race_id": r, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT",
                  "combination": "1", "payout_per_eur": 2.0},
                 {"race_id": r, "pool": "INTERNET", "bet_type": "SIMPLE_PLACE",
                  "combination": "2", "payout_per_eur": 1.4 + 0.1 * (r % 2)}]
    res = {x["rule"]: x for x in evaluate_frozen_rules(pd.DataFrame(rows), pd.DataFrame(pays),
                                                       "lgbm_market")}
    r5 = res["2e du modèle, placé, réunions R5+"]
    assert r5["n_bets"] == 120 and abs(r5["roi"] - 0.45) < 1e-9 and r5["confirmed"]
    assert res["4e du modèle, placé, quinté"]["n_bets"] == 0
    assert not res["4e du modèle, placé, quinté"]["confirmed"]
