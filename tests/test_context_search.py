"""Recherche par contexte : contextes, sélections, protocole (hasard et effet planté)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import context_search as cs


def _frame(n_days: int, plant: bool, seed: int = 0) -> pd.DataFrame:
    from tests.test_grand_search import _data

    from horse_intelligence.evaluation.profile_search import prepare

    pred, pays = _data(n_days, seed)
    f = prepare(pred, pays, "model")
    f = f[f["profit_gagnant"].notna()].copy()
    rid = f["race_id"].to_numpy()
    f["racecourse"] = np.where(rid % 3 == 0, "PLANTE", np.where(rid % 3 == 1, "AUTRE", "PETIT"))
    f["audience"] = np.where(rid % 2 == 0, "NATIONAL", "REGIONAL")
    f["label"] = "PRIX"
    f["conditions_text"] = np.where(f["discipline"] == "trot_attele", "Course G", "")
    f["distance_m"] = 2100
    f["prize_eur"] = 12000.0
    f["sex_condition"] = None
    if plant:
        rng = np.random.default_rng(1)
        m = ((f["racecourse"] == "PLANTE") & f["rank"].between(4, 6)
             & (f["mkt_odds"] > 9) & (f["mkt_odds"] <= 20)).to_numpy()
        f.loc[m, "profit_gagnant"] = np.where(rng.random(m.sum()) < 0.4, 3.0, -1.0)
    return f


def test_contexts_and_selections() -> None:
    f = cs.with_context(_frame(40, plant=False))
    in_a = np.ones(len(f), bool)
    ctx = cs.contexts(f, in_a)
    assert {"réunion nationale", "réunion régionale", "catégorie Trot G",
            "samedi ou dimanche", "lundi à vendredi"} <= set(ctx)
    assert ctx["catégorie Trot G"].sum() == (f["discipline"] == "trot_attele").sum()
    # hippodromes choisis sur la période A seulement
    none_in_a = cs.contexts(f, np.zeros(len(f), bool))
    assert not any(k.startswith("hippodrome") for k in none_in_a)
    assert len(cs.selections()) == 64
    assert cs._dist_label("plat", 0, 1400) == "plat, 1400 m et moins"
    assert cs._prize_label(50000.0, 1e12) == "allocation > 50 k€"


def test_fair_market_fails_and_planted_effect_is_found() -> None:
    fair = cs.run(_frame(120, plant=False), "model")
    assert fair["n_strategies"] == 64 * fair["n_contexts"]
    assert fair["success"] is False
    planted = cs.run(_frame(400, plant=True), "model")
    assert planted["success"] is True
    assert any("hippodrome PLANTE" in r["label"] and "4e à 6e" in r["label"]
               and "cote 9 à 20" in r["label"] for r in planted["finalists"])


def test_context_search_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_context_search
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_context_search(repo.engine)
    assert rep["n_races"] > 0 and isinstance(rep["success"], bool)
    assert rep["n_strategies"] == 64 * rep["n_contexts"]
