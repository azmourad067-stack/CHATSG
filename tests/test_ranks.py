"""Stratégies par rang : sélection du k-ième cheval, sous-ensemble quinté, règlement."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from horse_intelligence.evaluation.rank_strategies import rank_strategy_table, robust_winners


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, pays = [], []
    for r in range(40):
        day = date(2026, 1, 1) + timedelta(days=r)
        # le marché classe 1 > 2 > 3 ; le modèle inverse 1 et 2
        for num, mkt, mod in ((1, 0.5, 0.3), (2, 0.3, 0.5), (3, 0.2, 0.2)):
            preds.append({"race_id": r, "race_date": day, "num_pmu": num, "mkt_prob": mkt,
                          "m": mod})
        pays += [{"race_id": r, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT",
                  "combination": "2", "payout_per_eur": 3.0},
                 {"race_id": r, "pool": "INTERNET", "bet_type": "SIMPLE_PLACE",
                  "combination": "2", "payout_per_eur": 1.5}]
        if r % 2 == 0:
            pays.append({"race_id": r, "pool": "INTERNET", "bet_type": "QUINTE_PLUS",
                         "combination": "2-1-3", "payout_per_eur": 10.0})
    return pd.DataFrame(preds), pd.DataFrame(pays)


def test_rank_rules_pick_the_right_horse_and_subset() -> None:
    pred, pay = _data()
    rows = {r["rule"]: r for r in rank_strategy_table(pred, pay, "m")}
    assert rows["1er du modèle, gagnant, toutes"]["all"]["roi"] == 2.0     # cheval 2 gagne à 3 €
    assert rows["2e du marché, gagnant, toutes"]["all"]["roi"] == 2.0
    assert rows["1er du marché, gagnant, toutes"]["all"]["roi"] == -1.0
    assert rows["1er du modèle, placé, quinté"]["all"]["n_bets"] == 20
    assert len(rows) == 40
    winners = {r["rule"] for r in robust_winners(list(rows.values()))}
    assert "1er du modèle, gagnant, toutes" in winners
