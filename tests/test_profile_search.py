"""Recherche de profils de course : filtres, règlement, protocole 1re / 2nde moitié."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import profile_search as ps


def _frame(n_days: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, pays = [], []
    for d in range(n_days):
        for r in range(6):
            rid = d * 10 + r
            n = int(rng.integers(8, 16))
            p = rng.dirichlet(np.ones(n) * 2)
            win = int(rng.choice(n, p=p))
            disc = "trot_attele" if r % 2 else "plat"
            for i in range(n):
                rows.append({"race_id": rid, "race_date": date(2025, 1, 1) + timedelta(days=d),
                             "num_pmu": i + 1, "n_runners": float(n), "discipline": disc,
                             "mkt_prob": p[i], "mkt_odds": 0.85 / p[i],
                             "model": p[i] * np.exp(rng.normal(0, 0.1))})
            pays.append({"race_id": rid, "pool": "INTERNET", "bet_type": "SIMPLE_GAGNANT",
                         "combination": str(win + 1), "payout_per_eur": 0.85 / p[win]})
    return pd.DataFrame(rows), pd.DataFrame(pays)


def test_prepare_settles_win_and_skips_unsettled_place() -> None:
    frame, pays = _frame(5)
    f = ps.prepare(frame, pays, "model")
    assert f["profit_placé"].isna().all()  # aucun rapport placé collecté
    per_race = f.groupby("race_id")["profit_gagnant"].apply(lambda s: int((s > -1).sum()))
    assert (per_race == 1).all()
    assert (f.groupby("race_id")["rank"].min() == 1).all()


def test_profile_filters_and_example_label() -> None:
    frame, pays = _frame(10)
    f = ps.prepare(frame, pays, "model")
    s = ps.scan(f, [ps.EXAMPLE])
    expected = f[(f["rank"].between(2, 5)) & (f["mkt_odds"] > 4) & (f["mkt_odds"] <= 9)
                 & (f["discipline"] == "trot_attele") & f["n_runners"].between(10, 14)]
    assert s.loc[0, "bets"] == len(expected)
    assert ps.EXAMPLE.label() == ("2e à 5e du modèle, cote 4 à 9, trot attelé, "
                                  "10 à 14 partants, gagnant")
    assert len(ps.profile_grid()) == 7 * 6 * 5 * 4 * 2 * 2


def test_search_selects_on_first_half_only() -> None:
    frame, pays = _frame(80)
    f = ps.prepare(frame, pays, "model")
    rep = ps.search(f, min_bets=50)
    first, second, cut = ps.split_halves(f)
    assert first["race_date"].max() < second["race_date"].min() and cut == rep["cut"]
    assert 0 < len(rep["tested"]) <= ps.TOP_K and rep["alpha"] == 0.05 / len(rep["tested"])
    # les profils retenus sont les meilleurs ROI de la 1re moitié parmi les éligibles
    s1 = ps.scan(first, ps.profile_grid())
    best = s1[s1["bets"] >= 50].sort_values("roi", ascending=False)["roi"].iloc[0]
    assert rep["tested"][0]["first"]["roi"] == best
    # marché juste à 85 % : rien ne doit tenir sur la 2nde moitié
    assert rep["success"] is False


def test_profile_experiment_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_profile_search
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_profile_search(repo.engine)
    assert isinstance(rep["success"], bool)


def test_followed_profiles_are_the_outsiders_profile_and_its_place_variants() -> None:
    win, place, place_all = ps.FOLLOWED
    assert (win.rank, win.odds, win.discipline, win.runners) == (
        (4, 6), (20.0, 50.0), "trot_attele", (15, 99))
    assert win.bet == "gagnant" and win.edge
    assert (place.bet, place.edge, place_all.bet, place_all.edge) == ("placé", True,
                                                                      "placé", False)
