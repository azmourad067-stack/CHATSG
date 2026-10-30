from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from tests.synthetic import START_DAY, populate

from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.evaluation.walk_forward import make_folds, run_walk_forward
from horse_intelligence.models.registry import latest_model, load_artifact
from horse_intelligence.prediction.race_predictor import predict_race
from horse_intelligence.services.training import latest_report, train_and_evaluate


def test_folds_are_chronological_and_disjoint() -> None:
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(30)]
    folds = make_folds(dates, min_train_days=14, test_days=5, calib_days=4)
    assert len(folds) == 4
    for f in folds:
        assert max(f.train_dates) < min(f.calib_dates)
        assert max(f.calib_dates) < min(f.test_dates)
    tested = [d for f in folds for d in f.test_dates]
    assert tested == dates[14:]  # chaque date testée une seule fois, dans l'ordre


def test_walk_forward_never_trains_on_test_dates() -> None:
    seen: list[tuple[set[date], set[date]]] = []

    def spy(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        seen.append((set(base["race_date"]) | set(calib["race_date"]), set(test["race_date"])))
        return np.full(len(test), 0.5)

    frame = pd.DataFrame({"race_date": [date(2026, 1, 1) + timedelta(days=i // 2)
                                        for i in range(40)],
                          "race_id": range(40), "num_pmu": 1, "won": 1.0})
    run_walk_forward(frame, {"spy": spy}, min_train_days=8, test_days=3, calib_days=2)
    assert seen and all(not (train & test) and max(train) < min(test) for train, test in seen)


@pytest.fixture
def trained(repo: Repository, tmp_path) -> tuple[Repository, object]:  # type: ignore[no-untyped-def]
    populate(repo, days=30, with_results_last_day=False)
    rep = train_and_evaluate(repo.engine, tmp_path / "models", tmp_path / "reports",
                             min_train_days=14, test_days=5, calib_days=4)
    return repo, rep


def test_end_to_end_training(trained, tmp_path) -> None:  # type: ignore[no-untyped-def]
    repo, rep = trained
    assert rep.source_aggregates_audit["passed"]
    assert len(rep.folds) == 3
    m = rep.metrics_market_races
    # le signal synthétique est appris : mieux que l'uniforme
    assert m["lgbm_form"]["log_loss"] < m["uniform"]["log_loss"]
    assert m["form_logit"]["log_loss"] < m["uniform"]["log_loss"]
    # échantillon trop petit : aucun modèle candidat promu, la référence est active
    assert rep.registered["lgbm_market"]["status"] == "candidate"
    assert rep.registered["market_calibrated"]["status"] == "active"
    assert latest_report(tmp_path / "reports")["n_races"] == rep.n_races
    meta = latest_model(repo.engine, "lgbm_form")
    assert load_artifact(__import__("pathlib").Path(meta["artifact_uri"]),
                         meta["artifact_sha256"])["family"] == "lgbm_form"


def test_tampered_artifact_is_rejected(trained) -> None:  # type: ignore[no-untyped-def]
    repo, _ = trained
    from pathlib import Path
    meta = latest_model(repo.engine, "lgbm_form")
    path = Path(meta["artifact_uri"])
    path.write_text(path.read_text().replace('"lgbm_form"', '"autre"', 1))
    with pytest.raises(ValueError, match="empreinte"):
        load_artifact(path, meta["artifact_sha256"])


def test_predict_upcoming_race(trained) -> None:  # type: ignore[no-untyped-def]
    repo, _ = trained
    races = repo.races_on(START_DAY + timedelta(days=29))  # dernier jour : sans résultats
    race_id = int(races["race_id"].iloc[0])
    first = predict_race(repo.engine, race_id)
    again = predict_race(repo.engine, race_id)
    assert first.primary == "lgbm_market" and first.has_market
    t = first.table
    assert t["p_win"].sum() == pytest.approx(1.0)
    assert t["p_place"].sum() == pytest.approx(3.0)  # 8 partants → 3 places
    assert first.status in ("verified", "partial")
    pd.testing.assert_frame_equal(t, again.table)  # reproductible
    assert "non validé" in first.message


def test_predict_without_model(repo: Repository) -> None:
    populate(repo, days=2)
    race_id = int(repo.races_on(START_DAY)["race_id"].iloc[0])
    res = predict_race(repo.engine, race_id)
    assert res.table.empty and "Aucun modèle" in res.message


def test_training_requires_history(repo: Repository, tmp_path) -> None:  # type: ignore[no-untyped-def]
    populate(repo, days=5)
    with pytest.raises(ValueError, match="insuffisant"):
        train_and_evaluate(repo.engine, tmp_path / "m", tmp_path / "r", min_train_days=21)
