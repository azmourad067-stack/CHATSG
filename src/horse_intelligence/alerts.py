"""Alertes de pari par SMS (API « Notifications par SMS » de Free Mobile).

À chaque collecte (toutes les 15 min), les courses ouvertes aux paris en ligne (cotes
Internet publiées, toutes réunions) qui partent dans ALERT_WINDOW minutes sont examinées
avec la règle figée « Valeur 4e-6e », dans sa variante « tôt » (VALUE_46_EARLY, choisie par
l'utilisateur le 23/11/2026 à la place de l'alerte 10 à 25 min avant) : chaque cheval
retenu fait l'objet d'UN seul SMS, envoyé 45 à 60 minutes avant le départ. Au PMU, le gain
dépend du rapport final, pas de la cote au moment du pari.

Depuis le 24/11/2026, une 2e règle, « Petites courses, valeur » (SMALL_VALUE, réunions
régionales ou locales ou allocation ≤ 10 000 €), donne aussi un SMS, 13 à 28 minutes avant
le départ : c'est le moment qui correspond à son test (cotes de référence publiées par le
PMU 30 minutes avant le départ). Chaque cheval n'est signalé qu'une fois par règle.

Identifiants Free Mobile : secrets GitHub FREE_MOBILE_USER et FREE_MOBILE_KEY (jamais dans
le code). Sans eux, rien n'est envoyé.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from sqlalchemy import Engine, insert, select

from horse_intelligence.data.storage import database as db
from horse_intelligence.evaluation.forward_rules import (
    SMALL_VALUE,
    VALUE_46_EARLY,
    ValueRule,
    is_small_race,
    races_with_internet_odds,
    rule_picks,
)
from horse_intelligence.prediction.race_predictor import DayPrediction

FREE_MOBILE_URL = "https://smsapi.free-mobile.fr/sendmsg"
ALERT_WINDOW = (timedelta(minutes=45), timedelta(minutes=60))
RULE = VALUE_46_EARLY
RULE_NAME = "valeur_4e_6e_tot"


@dataclass(frozen=True)
class AlertRule:
    name: str                       # clé de déduplication (table alerts_sent)
    title: str                      # début du SMS
    rule: ValueRule
    window: tuple[timedelta, timedelta] = field(default=ALERT_WINDOW)


ALERT_RULES = (
    AlertRule(RULE_NAME, "Pari Valeur 4e-6e (1 h avant)", RULE, ALERT_WINDOW),
    AlertRule("petites_courses_valeur", "Pari Petites courses (valeur)", SMALL_VALUE,
              (timedelta(minutes=13), timedelta(minutes=28))),
)
RULE_MODEL = "lgbm_market"
PARIS = ZoneInfo("Europe/Paris")
FREE_MOBILE_ERRORS = {400: "paramètre manquant", 402: "trop de SMS envoyés en peu de temps",
                      403: "service non activé ou identifiants faux",
                      500: "erreur du serveur Free Mobile"}

Sender = Callable[[str], None]


class AlertError(RuntimeError):
    pass


def free_mobile_sender(user: str, key: str, timeout: float = 20.0) -> Sender:
    """Envoi d'un SMS sur la ligne Free Mobile de l'abonné (lui seul peut le recevoir)."""
    def send(msg: str) -> None:
        r = requests.post(FREE_MOBILE_URL, json={"user": user, "pass": key, "msg": msg},
                          timeout=timeout)
        if r.status_code != 200:
            raise AlertError(f"SMS non envoyé ({r.status_code} : "
                             f"{FREE_MOBILE_ERRORS.get(r.status_code, 'erreur inconnue')})")
    return send


def sender_from_env() -> Sender | None:
    user, key = os.environ.get("FREE_MOBILE_USER"), os.environ.get("FREE_MOBILE_KEY")
    return free_mobile_sender(user, key) if user and key else None


@dataclass
class Alert:
    race_id: int
    nums: list[int]
    text: str
    rule: str = RULE_NAME


def _paris(ts: datetime) -> str:
    return ts.astimezone(PARIS).strftime("%H:%M")


def format_alert(race: dict[str, Any], picks: pd.DataFrame, names: dict[int, str],
                 title: str = ALERT_RULES[0].title) -> str:
    start = race["scheduled_start_utc"]
    lines = [f"{title} - R{race['reunion_num']}C{race['course_num']} "
             f"{race['racecourse'] or ''} départ {_paris(start)}"]
    for p in picks.to_dict("records"):
        name = names.get(int(p["num_pmu"]), "")
        lines.append(f"Simple gagnant n°{int(p['num_pmu'])} {name} (cote {p['mkt_odds']:.1f}, "
                     f"{int(p['model_rank'])}e du modèle, espérance {100 * p['ev']:+.0f} %)")
    lines.append("Règle non validée : petite mise fixe. 09 74 75 13 13")
    return "\n".join(lines)


def due_alerts(engine: Engine, preds: dict[int, DayPrediction], now: datetime
               ) -> list[Alert]:
    """Alertes à envoyer maintenant, pour chaque règle de ALERT_RULES : courses ouvertes
    aux paris en ligne partant dans le créneau de la règle, chevaux retenus et pas encore
    signalés pour cette règle."""
    r, rc, sent = db.races, db.racecourses, db.alerts_sent
    alerts: list[Alert] = []
    for ar in ALERT_RULES:
        lo, hi = now + ar.window[0], now + ar.window[1]
        with engine.connect() as conn:
            races = conn.execute(
                select(r.c.race_id, r.c.reunion_num, r.c.course_num, r.c.scheduled_start_utc,
                       r.c.audience, r.c.prize_eur, rc.c.name.label("racecourse"))
                .join(rc, rc.c.racecourse_id == r.c.racecourse_id, isouter=True)
                .where(r.c.race_id.in_(list(preds)), r.c.scheduled_start_utc > lo,
                       r.c.scheduled_start_utc <= hi)).mappings().all()
            done = {(int(x.race_id), int(x.num_pmu)) for x in conn.execute(
                select(sent.c.race_id, sent.c.num_pmu).where(
                    sent.c.rule == ar.name, sent.c.race_id.in_(list(preds))))}
        if ar.rule.small_races:
            races = [x for x in races if is_small_race(x["audience"], x["prize_eur"])]
        online = races_with_internet_odds(engine, [int(x["race_id"]) for x in races])
        for race in races:
            rid = int(race["race_id"])
            pred = preds[rid]
            if rid not in online or pred.primary != RULE_MODEL or pred.table.empty:
                continue
            picks = rule_picks(pred.table, ar.rule)
            picks = picks[[(rid, int(n)) not in done for n in picks["num_pmu"]]]
            if picks.empty:
                continue
            names = _names(engine, rid)
            alerts.append(Alert(rid, [int(n) for n in picks["num_pmu"]],
                                format_alert(dict(race), picks, names, ar.title), ar.name))
    return alerts


def _names(engine: Engine, race_id: int) -> dict[int, str]:
    e, h = db.race_entries, db.horses
    with engine.connect() as conn:
        rows = conn.execute(select(e.c.num_pmu, h.c.display_name, e.c.observed_at)
                            .join(h, h.c.horse_id == e.c.horse_id)
                            .where(e.c.race_id == race_id)
                            .order_by(e.c.observed_at)).all()
    return {int(n): str(name) for n, name, _ in rows}


def send_alerts(engine: Engine, preds: dict[int, DayPrediction], sender: Sender,
                now: datetime | None = None) -> int:
    """Envoie les alertes dues et les note (une seule fois par cheval). Renvoie le nombre de
    SMS envoyés."""
    now = now or datetime.now(UTC)
    n = 0
    for alert in due_alerts(engine, preds, now):
        sender(alert.text)
        with engine.begin() as conn:
            conn.execute(insert(db.alerts_sent), [
                {"rule": alert.rule, "race_id": alert.race_id, "num_pmu": num,
                 "channel": "sms_free_mobile", "sent_at": now} for num in alert.nums])
        n += 1
    return n
