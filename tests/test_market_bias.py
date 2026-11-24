"""Biais du marché par contexte : carte, apprentissage des cases, protocole."""

from __future__ import annotations

import numpy as np

from horse_intelligence.evaluation import market_bias as mb


def test_ridge_attributes_the_effect_to_the_right_context() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    regional = rng.random(n) < 0.5
    weekday = rng.random(n) < 0.7
    # profit moyen −20 % partout, +30 % en réunion régionale ; le jour n'y est pour rien
    profit = np.where(rng.random(n) < np.where(regional, 0.13, 0.08), 9.0, -1.0)
    cells = [("tous", np.ones(n, bool)), ("régionale", regional),
             ("semaine", weekday), ("trop petite", np.arange(n) < 10)]
    pred, effects = mb.learn(cells, profit, np.ones(n, bool))
    assert effects[0][0] == "régionale"
    assert "trop petite" not in [e[0] for e in effects]
    assert (pred[regional] > 0).mean() > 0.95 and (pred[~regional] > 0).mean() < 0.05
    # rien appris : personne n'est joué
    pred0, eff0 = mb.learn(cells[3:], profit, np.ones(n, bool))
    assert eff0 == [] and np.isneginf(pred0).all()


def test_fair_market_fails_and_diffuse_bias_is_found() -> None:
    from tests.test_context_search import _frame

    fair = mb.run(_frame(120, plant=False), "model")
    assert fair["success"] is False
    assert {r["context"] for r in fair["map"]} >= {"réunion nationale", "réunion régionale"}
    f = _frame(400, plant=False)
    rng = np.random.default_rng(3)
    m = ((f["audience"] == "REGIONAL") & (f["mkt_odds"] > 9) & (f["mkt_odds"] <= 20)
         ).to_numpy()
    # +60 % en réunion régionale à cote 9-20 (au lieu de −15 % ailleurs)
    f.loc[m, "profit_gagnant"] = np.where(rng.random(m.sum()) < 0.16, 9.0, -1.0)
    planted = mb.run(f, "model")
    assert planted["success"] is True
    s1 = planted["strategies"]["S1"]["AB"]
    assert s1["C"]["roi"] > 0 and any("cote 9 à 20" in c["label"] and (
        "réunion régionale" in c["label"] or "Trot G" in c["label"]) for c in s1["top_cells"])


def test_market_bias_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_market_bias
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_market_bias(repo.engine)
    assert rep["n_races"] > 0 and isinstance(rep["success"], bool)
    assert set(rep["strategies"]) == {"S1", "S2", "S3", "S4"}
