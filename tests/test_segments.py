"""Analyse par segment : catégories de course, découpages et correction multiple."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.segment_analysis import (
    _bh,
    model_value_by_segment,
    race_group,
    segment_table,
)


def test_race_group_parsing() -> None:
    assert race_group("PRIX X (GR.A)", "PRIX X Course 3 Course F 15.000", "trot_attele") == \
        "Trot F"
    assert race_group("GRAND PRIX", "Groupe II - 4 ans", "plat") == "Groupe II"
    assert race_group("PRIX Y", "Pour chevaux ... valeur inférieure", "plat") == "Handicap"
    assert race_group("PRIX Z", "Course à réclamer", "plat") == "Réclamer"
    assert race_group("PRIX W", None, "obstacle") == "Galop (autre)"
    # un groupe cité dans les conditions d'engagement ne fait pas de la course un groupe
    assert race_group("PRIX V", "Pour chevaux de 3 ans, n'ayant pas gagné un Groupe I depuis",
                      "plat") == "Galop (autre)"


def test_bh_is_monotone_and_bounded() -> None:
    q = _bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert (q >= np.array([0.01, 0.04, 0.03, 0.5])).all() and (q <= 1).all()
    assert q[0] == 0.04  # 0.01 × 4 / 1


def _frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pays = [], []
    for r in range(200):
        day = date(2026, 1, 1) + timedelta(days=r // 2)
        for num, p, m in ((1, 0.45, 0.5), (2, 0.4, 0.3), (3, 0.15, 0.2)):
            rows.append({"race_id": r, "race_date": day, "num_pmu": num, "lgbm_market": p,
                         "market_calibrated": m, "mkt_prob": m, "won": float(num == 2),
                         "n_runners": 3, "distance_m": 2100, "discipline": "trot_attele",
                         "reunion_num": 1 + r % 3, "label": "PRIX", "conditions_text": "Course E"})
        pays.append({"race_id": r, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT",
                     "combination": "2", "payout_per_eur": 3.0 + r % 2})
    return pd.DataFrame(rows), pd.DataFrame(pays)


def test_segment_table_and_model_value() -> None:
    frame, pays = _frame()
    t = segment_table(frame, pays, "lgbm_market")
    rank2 = t[(t["dimension"] == "toutes") & (t["rang"] == "2") & (t["pari"] == "gagnant")]
    assert len(rank2) == 1 and abs(rank2["roi"].iloc[0] - 2.5) < 1e-9
    assert rank2["robuste"].iloc[0]
    assert set(t["dimension"]) >= {"discipline", "réunion", "groupe", "toutes"}
    v = model_value_by_segment(frame, "lgbm_market", "market_calibrated")
    assert (v["gain_logloss"] < 0).all()   # le modèle donne 0,40 au gagnant contre 0,30
