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
    rep = train_and_evaluate(repo.engine, min_train_days=14, test_days=5, calib_days=4,
                             n_members=3)
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
    # modèle placé enregistré, mais non promu sur un si petit échantillon
    assert rep.registered["place_market"]["status"] == "candidate"
    assert set(rep.place) == {"n_races", "vs_incumbent", "vs_market", "incumbent_vs_market"}
    assert latest_report(repo.engine)["n_races"] == rep.n_races
    meta = latest_model(repo.engine, "lgbm_form")
    assert meta["artifact_uri"].startswith("db://models/")
    assert load_artifact(repo.engine, meta["artifact_uri"],
                         meta["artifact_sha256"])["family"] == "lgbm_form"


def test_tampered_artifact_is_rejected(trained) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import update

    from horse_intelligence.data.storage import database as db

    repo, _ = trained
    meta = latest_model(repo.engine, "lgbm_form")
    key = meta["artifact_uri"].removeprefix("db://")
    with repo.engine.begin() as conn:
        conn.execute(update(db.stored_files).where(db.stored_files.c.key == key)
                     .values(content=b'{"family": "autre"}'))
    with pytest.raises(ValueError, match="empreinte"):
        load_artifact(repo.engine, meta["artifact_uri"], meta["artifact_sha256"])


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
    assert not first.in_sample  # dernier jour sans résultat : hors apprentissage
    assert first.uncertainty in ("exploitable", "incertaine", "hors distribution")
    assert t["p_low"].notna().all() and (t["p_low"] <= t["p_high"]).all()
    assert t["strengths"].str.len().sum() > 0  # explications produites
    assert (t["weak_signals"] == "").all()  # aucun pattern validé en base
    # contributions relatives : somme nulle dans la course (seules les différences comptent)
    from horse_intelligence.intelligence.explainability import relative_contributions
    from horse_intelligence.intelligence.feature_engineering import build_dataset, load_raw
    from horse_intelligence.models.lightgbm_model import LgbmRaceModel
    from horse_intelligence.models.registry import latest_model, load_artifact
    meta = latest_model(repo.engine, "lgbm_market")
    model = LgbmRaceModel.from_dict(load_artifact(repo.engine, meta["artifact_uri"]))
    frame = build_dataset(load_raw(repo.engine)).frame
    race = frame[frame["race_id"] == race_id]
    assert np.allclose(relative_contributions(model, race).sum(axis=0), 0.0)
    trained_race = int(repo.races_on(START_DAY + timedelta(days=3))["race_id"].iloc[0])
    assert predict_race(repo.engine, trained_race).in_sample


def test_place_model_used_only_once_validated(trained) -> None:  # type: ignore[no-untyped-def]
    from horse_intelligence.models.registry import activate_version

    repo, rep = trained
    race_id = int(repo.races_on(START_DAY + timedelta(days=29))["race_id"].iloc[0])
    before = predict_race(repo.engine, race_id)
    assert before.place_method == "harville"  # modèle placé seulement candidat
    activate_version(repo.engine, "place_market", rep.registered["place_market"]["version"])
    after = predict_race(repo.engine, race_id)
    assert after.place_method == "model"
    p = after.table["p_place"]
    assert p.sum() == pytest.approx(3.0, abs=1e-6) and p.between(0, 1).all()
    assert not np.allclose(p.sort_index(), before.table["p_place"].sort_index())
    pd.testing.assert_series_equal(after.table["p_win"], before.table["p_win"])


def test_predict_without_model(repo: Repository) -> None:
    populate(repo, days=2)
    race_id = int(repo.races_on(START_DAY)["race_id"].iloc[0])
    res = predict_race(repo.engine, race_id)
    assert res.table.empty and "Aucun modèle" in res.message


def test_training_requires_history(repo: Repository, tmp_path) -> None:  # type: ignore[no-untyped-def]
    populate(repo, days=5)
    with pytest.raises(ValueError, match="insuffisant"):
        train_and_evaluate(repo.engine, min_train_days=21)


def test_lab_end_to_end(trained, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from horse_intelligence.services.discovery import latest_lab_report, run_lab

    repo, _ = trained
    rep = run_lab(repo.engine)
    assert rep.n_tests > 0
    assert {"jockeys", "trainers"} <= set(rep.actors)
    assert rep.anomalies["examples"] and rep.drift["psi"]
    assert latest_lab_report(repo.engine)["n_tests"] == rep.n_tests


def test_validation_end_to_end(trained, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from horse_intelligence.services.validation import latest_validation, run_validation

    repo, _ = trained
    rep = run_validation(repo.engine)
    assert rep["races_with_payouts"] > 0
    assert all(c["passed"] for c in rep["leakage"]), rep["leakage"]
    assert rep["robustness"]["probabilities_valid"]
    for b in rep["betting"].values():
        assert b["selection_period"][1] < b["test_period"][0]
    # repère : parier tous les favoris d'un marché qui prélève ~15 % perd de l'argent
    fav = rep["references_all_period"][0]
    assert fav["n_bets"] > 0 and fav["roi"] < 0.1
    assert latest_validation(repo.engine)["races"] == rep["races"]


def test_predict_day_matches_predict_race(trained) -> None:  # type: ignore[no-untyped-def]
    from horse_intelligence.prediction.race_predictor import (
        build_frame,
        load_models,
        models_use_source_aggregates,
        predict_day,
    )

    repo, _ = trained
    day = START_DAY + timedelta(days=29)
    ids = [int(x) for x in repo.races_on(day)["race_id"]]
    models = load_models(repo.engine)
    frame = build_frame(repo.engine, day, models_use_source_aggregates(models))
    batch = predict_day(repo.engine, ids, frame, models)
    assert set(batch) == set(ids)
    for rid in ids:
        one = predict_race(repo.engine, rid, frame=frame, models=models)
        got = batch[rid]
        assert got.primary == one.primary
        cols = ["num_pmu", "horse_id", "p_win", "rank", "mkt_odds"]
        pd.testing.assert_frame_equal(
            got.table[cols].sort_values("num_pmu").reset_index(drop=True),
            one.table[cols].sort_values("num_pmu").reset_index(drop=True),
            check_dtype=False)
    # partants de toutes les courses en une requête = partants course par course
    grouped = repo.entries_for_races(ids)
    for rid in ids:
        one_e = repo.entries_as_of(rid)
        assert grouped[grouped["race_id"] == rid]["num_pmu"].tolist() == \
            one_e["num_pmu"].tolist()


def test_precomputed_day_predictions_round_trip(trained) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from horse_intelligence.prediction import day_store
    from horse_intelligence.prediction.forward import precompute_day
    from horse_intelligence.prediction.race_predictor import (
        build_frame,
        load_models,
        models_use_source_aggregates,
        predict_day,
    )

    repo, _ = trained
    day = START_DAY + timedelta(days=29)
    ids = [int(x) for x in repo.races_on(day)["race_id"]]
    before_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    # plus aucune course à courir : rien n'est calculé
    assert precompute_day(repo.engine, day, now=datetime(2100, 1, 1, tzinfo=UTC)) == {}
    assert len(precompute_day(repo.engine, day, now=before_start)) == len(ids)
    key = day_store.data_key(repo.engine)
    stored = day_store.load(repo.engine, ids, key)
    assert stored is not None and set(stored) == set(ids)
    models = load_models(repo.engine)
    live = predict_day(repo.engine, ids, build_frame(repo.engine, day,
                                                     models_use_source_aggregates(models)),
                       models)
    cols = ["num_pmu", "horse_id", "p_win", "rank", "mkt_odds", "mkt_prob"]
    for rid in ids:
        assert stored[rid].primary == live[rid].primary
        pd.testing.assert_frame_equal(stored[rid].table[cols], live[rid].table[cols],
                                      check_dtype=False)
    # autre clé (nouvelle donnée, nouveau modèle) ou course manquante : jamais utilisé
    assert day_store.load(repo.engine, ids, key + "x") is None
    assert day_store.load(repo.engine, [*ids, 999999], key) is None
    # nouveau calcul : remplace, ne duplique pas
    assert len(precompute_day(repo.engine, day, now=before_start)) == len(ids)
    assert day_store.load(repo.engine, ids, key) is not None
