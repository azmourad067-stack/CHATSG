"""Alertes SMS : créneau avant départ, un seul message par cheval, erreurs Free Mobile."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import select, update

from horse_intelligence import alerts
from horse_intelligence.data.storage import database as db
from horse_intelligence.prediction.race_predictor import DayPrediction

P = [0.25, 0.18, 0.12, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.04]
ODDS = [3.0, 5.0, 8.0, 12.0, 20.0, 9.0, 15.0, 18.0, 30.0, 40.0]  # retient le n°5


def _table() -> pd.DataFrame:
    t = pd.DataFrame({"num_pmu": range(1, 11), "horse_id": range(1, 11), "p_win": P,
                      "mkt_odds": ODDS, "mkt_prob": P})
    t["rank"] = t["p_win"].rank(ascending=False, method="min")
    return t


@pytest.fixture
def setup(repo):  # type: ignore[no-untyped-def]
    from tests.synthetic import populate

    populate(repo, days=2)
    with repo.engine.connect() as conn:
        rows = conn.execute(select(db.races.c.race_id, db.races.c.scheduled_start_utc)
                            .order_by(db.races.c.race_id).limit(3)).all()
    preds = {int(r.race_id): DayPrediction(int(r.race_id), "lgbm_market", _table())
             for r in rows}
    return repo, rows, preds


def test_one_sms_per_horse_inside_window_only(setup) -> None:  # type: ignore[no-untyped-def]
    repo, rows, preds = setup
    first = rows[0]
    now = first.scheduled_start_utc - timedelta(minutes=50)
    sent: list[str] = []
    n = alerts.send_alerts(repo.engine, {first.race_id: preds[first.race_id]}, sent.append,
                           now)
    assert n == 1 and "n°5" in sent[0] and "Valeur 4e-6e (1 h avant)" in sent[0]
    assert "09 74 75 13 13" in sent[0]
    # collecte suivante, toujours dans le créneau : pas de doublon
    assert alerts.send_alerts(repo.engine, preds, sent.append, now + timedelta(minutes=5)) == 0
    # créneau 45 à 60 min avant : rien à 70 min, ni à 40 ou 5 min du départ
    with repo.engine.begin() as conn:
        conn.execute(db.alerts_sent.delete())
    one = {first.race_id: preds[first.race_id]}
    for minutes in (70, 40, 5):
        assert alerts.due_alerts(repo.engine, one,
                                 first.scheduled_start_utc - timedelta(minutes=minutes)) == []
    assert len(alerts.due_alerts(repo.engine, one,
                                 first.scheduled_start_utc - timedelta(minutes=59))) == 1


def test_regional_races_alerted_only_when_online_betting_is_open(setup) -> None:  # type: ignore[no-untyped-def]
    repo, rows, preds = setup
    first = rows[0]
    now = first.scheduled_start_utc - timedelta(minutes=50)
    one = {first.race_id: preds[first.race_id]}
    with repo.engine.begin() as conn:
        conn.execute(update(db.races).where(db.races.c.race_id == first.race_id)
                     .values(audience="REGIONAL"))
    # réunion régionale avec cotes Internet : la règle s'applique, comme dans les tests
    assert [a.race_id for a in alerts.due_alerts(repo.engine, one, now)] == [first.race_id]
    # sans cote Internet (pas de pari en ligne) : aucune alerte
    with repo.engine.begin() as conn:
        conn.execute(update(db.odds_snapshots)
                     .where(db.odds_snapshots.c.race_id == first.race_id)
                     .values(pool="OFFLINE"))
    assert alerts.due_alerts(repo.engine, one, now) == []
    no_market_model = {first.race_id: DayPrediction(first.race_id, "lgbm_form", _table())}
    assert alerts.due_alerts(repo.engine, no_market_model, now) == []


def test_alerts_follow_the_early_rule() -> None:
    from horse_intelligence.evaluation.forward_rules import VALUE_46, VALUE_46_EARLY

    assert alerts.RULE is VALUE_46_EARLY and VALUE_46_EARLY.lead_minutes == (45, 60)
    assert (timedelta(minutes=45), timedelta(minutes=60)) == alerts.ALERT_WINDOW
    # mêmes critères que la règle d'origine, seul le moment de la décision change
    for f in ("ranks", "odds", "runners", "min_ev", "payout_factor", "bet_type"):
        assert getattr(VALUE_46_EARLY, f) == getattr(VALUE_46, f)


def test_small_races_rule_alerts_13_to_28_minutes_before(setup) -> None:  # type: ignore[no-untyped-def]
    repo, rows, preds = setup
    first = rows[0]
    one = {first.race_id: preds[first.race_id]}
    small = next(a for a in alerts.ALERT_RULES if a.name == "petites_courses_valeur")
    assert small.window == (timedelta(minutes=13), timedelta(minutes=28))
    at = first.scheduled_start_utc - timedelta(minutes=20)
    # course nationale à 20 000 € : pas une petite course, rien à 20 min
    assert alerts.due_alerts(repo.engine, one, at) == []
    with repo.engine.begin() as conn:
        conn.execute(update(db.races).where(db.races.c.race_id == first.race_id)
                     .values(audience="LOCAL"))
    due = alerts.due_alerts(repo.engine, one, at)
    assert [a.rule for a in due] == ["petites_courses_valeur"]
    assert due[0].text.startswith("Pari Petites courses (valeur)")
    sent: list[str] = []
    assert alerts.send_alerts(repo.engine, one, sent.append, at) == 1
    assert alerts.send_alerts(repo.engine, one, sent.append, at + timedelta(minutes=5)) == 0
    # les deux règles sont indépendantes : la règle « 1 h avant » alerte aussi à 50 min
    assert [a.rule for a in alerts.due_alerts(
        repo.engine, one, first.scheduled_start_utc - timedelta(minutes=50))] == [
        alerts.RULE_NAME]


def test_free_mobile_sender(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = []

    def fake_post(url, json, timeout):  # type: ignore[no-untyped-def]
        calls.append((url, json))
        return SimpleNamespace(status_code=200 if json["pass"] == "bonne" else 403)

    monkeypatch.setattr(alerts.requests, "post", fake_post)
    alerts.free_mobile_sender("12345678", "bonne")("Bonjour")
    assert calls[0] == (alerts.FREE_MOBILE_URL,
                        {"user": "12345678", "pass": "bonne", "msg": "Bonjour"})
    with pytest.raises(alerts.AlertError, match="identifiants"):
        alerts.free_mobile_sender("12345678", "fausse")("Bonjour")
    monkeypatch.delenv("FREE_MOBILE_USER", raising=False)
    assert alerts.sender_from_env() is None
