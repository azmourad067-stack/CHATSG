from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.intelligence.explainability import strengths_weaknesses
from horse_intelligence.intelligence.uncertainty import OodScorer, race_status, resample_races


def test_resample_keeps_whole_races() -> None:
    f = pd.DataFrame({"race_id": [1, 1, 1, 2, 2, 3], "x": range(6)})
    out = resample_races(f, np.random.default_rng(0))
    assert out["race_id"].nunique() == 3
    sizes = out.groupby("race_id").size().sort_values().tolist()
    assert all(s in (1, 2, 3) for s in sizes)
    # chaque course tirée est complète (mêmes valeurs x qu'une course d'origine)
    origin = {tuple(g["x"]) for _, g in f.groupby("race_id")}
    assert {tuple(g["x"]) for _, g in out.groupby("race_id")} <= origin


def test_ood_scorer_flags_extremes_and_roundtrips() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    sc = OodScorer(["a", "b"]).fit(train)
    s = sc.runner_scores(pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, np.nan]}))
    assert s[1] > 5 * max(s[0], 0.1)
    s2 = OodScorer.from_dict(sc.to_dict()).runner_scores(pd.DataFrame({"a": [10.0], "b": [0.0]}))
    assert s2[0] > 3
    assert sc.top_deviations(pd.Series({"a": 10.0, "b": 0.1}))[0][0] == "a"


def test_race_status_rules() -> None:
    th = {"ood_race": 2.0, "rel_width": 0.5, "disagreement": 0.15}
    p = np.array([0.5, 0.3, 0.2])
    ok, why = race_status(p, p * 0.9, p * 1.1, np.zeros(3), p, th)
    assert ok == "exploitable" and not why
    wide, why = race_status(p, p * 0.5, p * 1.5, np.zeros(3), p, th)
    assert wide == "incertaine" and "instables" in why[0]
    dis, _ = race_status(p, p * 0.9, p * 1.1, np.zeros(3), np.array([0.2, 0.3, 0.5]), th)
    assert dis == "incertaine"
    ood, _ = race_status(p, p * 0.9, p * 1.1, np.full(3, 5.0), p, th)
    assert ood == "hors distribution"


def test_strengths_weaknesses_labels() -> None:
    c = pd.Series({"jockey_win_rate": 0.4, "mus_mean_pos_3": -0.3, "age": 0.01})
    s, w = strengths_weaknesses(c)
    assert s == ["réussite du jockey/driver"] and w == ["forme sur 3 courses"]
