"""Grande recherche : formules combinées, règlement, protocole A / B / C."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from horse_intelligence.evaluation import grand_search as gs
from horse_intelligence.evaluation.profile_search import prepare


def _data(n_days: int = 80, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marché « juste » à 85 % : aucune stratégie ne doit tenir sur la période C."""
    rng = np.random.default_rng(seed)
    rows, pays = [], []
    for d in range(n_days):
        for r in range(6):
            rid = d * 10 + r
            n = int(rng.integers(8, 16))
            p = rng.dirichlet(np.ones(n) * 2)
            order = list(rng.choice(n, size=n, replace=False, p=p))
            for i in range(n):
                rows.append({"race_id": rid, "race_date": date(2025, 1, 1) + timedelta(days=d),
                             "num_pmu": i + 1, "n_runners": float(n),
                             "discipline": "trot_attele" if r % 2 else "plat",
                             "mkt_prob": p[i], "mkt_odds": 0.85 / p[i],
                             "model": p[i] * np.exp(rng.normal(0, 0.1))})
            a, b, c = (int(x) + 1 for x in order[:3])
            pays += [{"race_id": rid, "bet_type": "SIMPLE_GAGNANT", "combination": str(a),
                      "payout_per_eur": 0.85 / p[a - 1]},
                     {"race_id": rid, "bet_type": "COUPLE_GAGNANT",
                      "combination": f"{a}-{b}", "payout_per_eur": 20.0},
                     {"race_id": rid, "bet_type": "TRIO", "combination": f"{a}-{b}-{c}",
                      "payout_per_eur": 60.0}]
    return pd.DataFrame(rows), pd.DataFrame(pays).assign(pool="INTERNET")


def test_combo_formulas_ticket_counts() -> None:
    r = [7, 3, 9, 1, 5, 2]
    counts = {name: len(build(r)) for name, (_, build) in gs.COMBO_FORMULAS.items()}
    assert counts["couplé gagnant 1-2"] == 1 and counts["couplé gagnant 1 avec 2, 3, 4"] == 3
    assert counts["trio 1-2-3-4 combiné"] == 4 and counts["trio 1 avec 2 à 5 combiné"] == 6
    assert counts["couplé placé 1 avec 2 à 5"] == 4 and counts["trio 1-2 avec 3 à 6"] == 4


def test_combo_races_settle_on_paid_combination() -> None:
    pred, pays = _data(3)
    races = gs.combo_races(pred, pays, "model")
    race = races.iloc[0]
    g = pred[pred["race_id"] == race["race_id"]]
    top = g.sort_values("model", ascending=False)["num_pmu"].tolist()
    paid = pays[(pays["race_id"] == race["race_id"]) & (pays["bet_type"] == "TRIO")]
    winners = {int(x) for x in paid["combination"].iloc[0].split("-")}
    expected = 60.0 if set(top[:3]) == winners else 0.0
    assert race["modèle|trio 1-2-3|stake"] == 1 and race["modèle|trio 1-2-3|ret"] == expected
    assert np.isnan(race["modèle|2 sur 4 1-2|stake"])  # pari non collecté : non joué


def test_protocol_and_no_false_success() -> None:
    pred, pays = _data(120)
    cuts = gs.cuts_of(pred["race_date"])
    f = prepare(pred, pays, "model")
    s, c, g = gs.simple_scan(f, "model", cuts)
    assert len(s) == 7 * 6 * 5 * 4 * 4 * 2
    races = gs.combo_races(pred, pays, "model")
    s2, c2, g2 = gs.combo_scan(races, cuts)
    assert len(s2) == 2 * len(gs.COMBO_FORMULAS) * 5 * 4 * 4
    sel = gs.select([*s, *s2], np.vstack([c, c2]), np.vstack([g, g2]))
    roi = sel["roi"]
    # découverte classée sur A seulement, finalistes positives en B
    assert all(roi[i, 1] > 0 for i in sel["final"])
    disc_roi = [roi[i, 0] for i in sel["discovery"]]
    assert disc_roi == sorted(disc_roi, reverse=True)
    # le détail de chaque stratégie (échantillon des deux familles, tous filtres) retrouve
    # exactement les mises et profits du balayage, période par période
    strategies = [*s, *s2]
    counts, gains = np.vstack([c, c2]), np.vstack([g, g2])
    rng = np.random.default_rng(1)
    sample = [*rng.choice(len(s), 150, replace=False),
              *(len(s) + rng.choice(len(s2), 80, replace=False))]
    for i in sample:
        st = strategies[i]
        b = gs.simple_bets(f, "model", st) if st.family == "simple" \
            else gs.combo_bets(races, st)
        per = gs.periods(b["race_date"], cuts) if not b.empty else np.array([], int)
        for k in range(3):
            assert b.loc[per == k, "stake"].sum() == counts[i, k], st.label
            assert np.isclose(b.loc[per == k, "profit"].sum(), gains[i, k]), st.label


def test_grand_search_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_grand_search
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_grand_search(repo.engine)
    assert isinstance(rep["success"], bool)


def test_value46_bet_types_end_to_end(repo) -> None:  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    from horse_intelligence.services.experiment import run_value46_bet_types
    from horse_intelligence.services.training import train_and_evaluate

    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=1)
    rep = run_value46_bet_types(repo.engine)
    if rep["n_races"]:
        assert set(rep["results"]) == {"gagnant", "placé", "gagnant + placé"}
        g, p, both = (rep["results"][k]["total"] for k in ("gagnant", "placé",
                                                           "gagnant + placé"))
        if g.get("bets"):
            assert both["bets"] == g["bets"] + p["bets"]  # mêmes chevaux, deux mises
