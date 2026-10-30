"""Générateur de données SYNTHÉTIQUES (tests uniquement) : chevaux à capacité latente,
arrivées tirées selon Plackett-Luce, musique et cotes corrélées à cette capacité."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np

from horse_intelligence.data.processors.validation import validate_snapshot
from horse_intelligence.data.schemas.domain import (
    EntryRecord,
    OddsRecord,
    PayoutRecord,
    RaceRecord,
    RaceSnapshot,
    ResultRecord,
)
from horse_intelligence.data.storage.repositories import Repository

START_DAY = date(2026, 6, 1)


def populate(repo: Repository, days: int = 30, races_per_day: int = 6, runners: int = 8,
             seed: int = 7, with_results_last_day: bool = True) -> None:
    rng = np.random.default_rng(seed)
    n_horses = 160
    ability = rng.normal(0, 1, n_horses)
    jockey_skill = rng.normal(0, 0.5, 25)
    for d in range(days):
        day = START_DAY + timedelta(days=d)
        for c in range(1, races_per_day + 1):
            start = datetime(day.year, day.month, day.day, 12, 0, tzinfo=UTC) + timedelta(
                minutes=30 * c)
            race = RaceRecord(day, 1, c, "SYN", "Synthétique", "FRA", f"PRIX {c}",
                              "trot_attele", "ATTELE", "TROT_ATTELE", 2700, None, 20000.0,
                              runners, None, None, start, "ARRIVEE_DEFINITIVE_COMPLETE", True,
                              None, None)
            horses = rng.choice(n_horses, size=runners, replace=False)
            jockeys = rng.choice(len(jockey_skill), size=runners, replace=False)
            strength = ability[horses] + jockey_skill[jockeys]
            p = np.exp(strength) / np.exp(strength).sum()
            # ordre d'arrivée Plackett-Luce (Gumbel)
            order = np.argsort(-(strength + rng.gumbel(size=runners)))
            final = with_results_last_day or d < days - 1
            observed = start + timedelta(hours=1) if final else start - timedelta(minutes=20)
            snap = RaceSnapshot(race, observed)
            for i, h in enumerate(horses):
                recent = np.clip(np.round(4.5 - 1.5 * ability[h] + rng.normal(0, 2, 6)), 1, 9)
                musique = "".join(f"{int(x)}a" for x in recent)
                snap.entries.append(EntryRecord(
                    i + 1, f"CHEVAL {h}", f"CHEVAL {h}", 2020, None, 6, "HONGRES",
                    f"J. JOCKEY{jockeys[i]}", f"E. TRAINER{h % 30}", None, None, None, None,
                    2700, None, None, "PARTANT", musique, 20, int(max(0, 3 + ability[h] * 2)),
                    8, None))
                noisy = p[i] * np.exp(rng.normal(0, 0.25))
                snap.odds.append(OddsRecord(i + 1, "SIMPLE_GAGNANT", "reference",
                                            float(max(1.1, 0.85 / noisy)),
                                            start - timedelta(minutes=30), "INTERNET"))
            if final:
                for pos, idx in enumerate(order, start=1):
                    snap.results.append(ResultRecord(int(idx) + 1, pos, None, None, None))
                # rapports synthétiques : pari mutuel avec prélèvement d'environ 15 %
                win = int(order[0])
                snap.payouts.append(PayoutRecord("SIMPLE_GAGNANT", str(win + 1),
                                                 round(0.85 / p[win], 2), "INTERNET"))
                for idx in order[:3]:
                    snap.payouts.append(PayoutRecord(
                        "SIMPLE_PLACE", str(int(idx) + 1),
                        round(max(1.1, 0.85 / min(0.95, 3 * p[idx])), 2), "INTERNET"))
            race_id = repo.upsert_race(race, observed, "pmu", None)
            repo.save_snapshot(race_id, snap, None)
            repo.save_quality_report(validate_snapshot(snap))
