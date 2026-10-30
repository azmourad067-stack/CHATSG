"""Phase 6 : stockage en base, collecte légère, prédictions figées et suivi réel."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from tests.conftest import RACE_DAY, START
from tests.synthetic import START_DAY, populate
from tests.test_no_data_leakage import FakePmu

from horse_intelligence.data.storage import blob_store
from horse_intelligence.data.storage import database as db
from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.monitoring.data_monitor import collection_health
from horse_intelligence.monitoring.model_monitor import live_tracking
from horse_intelligence.prediction.forward import record_upcoming
from horse_intelligence.services.ingestion import ingest_day
from horse_intelligence.services.training import train_and_evaluate


def test_blob_store_roundtrip_tamper_and_prune(repo: Repository) -> None:
    e = repo.engine
    blob_store.put_json(e, "reports/x-1.json", {"a": 1})
    assert blob_store.get_json(e, "reports/x-1.json") == {"a": 1}
    assert blob_store.get(e, "absent") is None
    with e.begin() as conn:
        conn.execute(update(db.stored_files).where(db.stored_files.c.key == "reports/x-1.json")
                     .values(content=b'{"a": 2}'))
    with pytest.raises(ValueError, match="empreinte"):
        blob_store.get(e, "reports/x-1.json")
    for i in range(5):
        blob_store.put_json(e, f"models/m-{i}.json", {"i": i})
    blob_store.put_json(e, "models/m-latest.json", {})
    removed = blob_store.prune(e, "models/m-", keep=2, protected={"models/m-0.json"})
    assert removed == 2
    assert blob_store.get_json(e, "models/m-0.json") == {"i": 0}  # protégé (modèle actif)
    assert blob_store.get_json(e, "models/m-1.json") is None
    assert blob_store.get_json(e, "models/m-latest.json") == {}


def test_window_collection_only_touches_relevant_races(repo: Repository) -> None:
    # 3 h avant : aucune course dans la fenêtre de 75 min
    early = ingest_day(RACE_DAY, FakePmu(START - timedelta(hours=3), False), repo,
                       window=timedelta(minutes=75), now=START - timedelta(hours=3))
    assert early.races_seen == 0
    # 30 min avant : R1C1 (départ) et R1C2 (départ + 30 min) sont dans la fenêtre
    before = ingest_day(RACE_DAY, FakePmu(START - timedelta(minutes=30), False), repo,
                        window=timedelta(minutes=75), now=START - timedelta(minutes=30))
    assert before.races_seen == 2
    # 2 h après : seule R1C1 (terminée, sans résultat en base) est recollectée
    after = ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), repo,
                       window=timedelta(minutes=75), now=START + timedelta(hours=2))
    assert after.races_seen == 1
    assert repo.has_results(RACE_DAY, 1, 1)
    again = ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=3), True), repo,
                       window=timedelta(minutes=75), now=START + timedelta(hours=3))
    assert again.races_seen == 0


def test_forward_predictions_are_frozen_then_settled(repo: Repository) -> None:
    populate(repo, days=24)
    train_and_evaluate(repo.engine, min_train_days=12, test_days=4, calib_days=4, n_members=2)
    last_day = START_DAY + timedelta(days=23)
    now = datetime(last_day.year, last_day.month, last_day.day, 12, 0, tzinfo=UTC)
    saved = record_upcoming(repo.engine, timedelta(hours=4), now=now)
    assert saved == 6  # les 6 courses du dernier jour partent entre 12 h 30 et 15 h
    live = live_tracking(repo.engine)
    assert live["races"] == 6
    assert "vs_market" in live and live["model"]["log_loss"] > 0
    # une prédiction calculée APRÈS le départ n'est jamais comptée
    late = record_upcoming(repo.engine, timedelta(hours=4), now=now + timedelta(hours=5))
    assert late == 0


def test_collection_health(repo: Repository) -> None:
    assert collection_health(repo.engine)["stale"]
    run = repo.start_run(RACE_DAY)
    repo.finish_run(run, "success", 1, 1, [])
    assert not collection_health(repo.engine)["stale"]


def test_rollback_and_read_only(repo: Repository, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from horse_intelligence.models.registry import activate_version, latest_model, list_versions

    populate(repo, days=20)
    train_and_evaluate(repo.engine, min_train_days=10, test_days=4, calib_days=3, n_members=1)
    first = latest_model(repo.engine, "lgbm_market")["version"]
    import time
    time.sleep(1.1)  # versions horodatées à la seconde
    train_and_evaluate(repo.engine, min_train_days=10, test_days=4, calib_days=3, n_members=1)
    assert len(list_versions(repo.engine, "lgbm_market")) == 2
    activate_version(repo.engine, "lgbm_market", first)
    assert latest_model(repo.engine, "lgbm_market")["version"] == first
    assert latest_model(repo.engine, "lgbm_market")["status"] == "active"
    with pytest.raises(ValueError, match="introuvable"):
        activate_version(repo.engine, "lgbm_market", "inexistante")

    from horse_intelligence.config import settings as settings_module
    monkeypatch.setenv("HI_READ_ONLY", "1")
    assert settings_module.build_settings().read_only
