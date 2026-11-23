"""Signaux d'information affichés dans l'interface (pas des règles de pari validées).

« Déferré des 4 pour la 1re fois » (trot) : le cheval court déferré des quatre pieds alors
qu'il ne l'était pas à sa dernière course connue. Même définition que la variable
`shoes_d4_first` (feature_engineering._gear) et que l'angle testé le 22/11/2026 :
+0,2 % sur deux ans (16 352 paris, contre −21,3 % pour l'ensemble des partants), mais
−11,7 % sur les six derniers mois : signal réel, trop faible pour être rentable.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd
from sqlalchemy import Engine, select

from horse_intelligence.data.storage import database as db

D4 = "DEFERRE_ANTERIEURS_POSTERIEURS"


def _latest(entries: pd.DataFrame) -> pd.DataFrame:
    """Dernière version connue de chaque partant (course, numéro)."""
    return entries.sort_values("observed_at").drop_duplicates(["race_id", "num_pmu"],
                                                               keep="last")


def first_time_d4(engine: Engine, race_ids: Sequence[int], day: date) -> set[tuple[int, int]]:
    """(course, numéro) des trotteurs déferrés des 4 pour la 1re fois dans ces courses."""
    if not race_ids:
        return set()
    e, r = db.race_entries, db.races
    cols = [e.c.race_id, e.c.num_pmu, e.c.horse_id, e.c.shoes, e.c.status, e.c.observed_at]
    with engine.connect() as conn:
        today = pd.read_sql(select(*cols, r.c.discipline).join(r, r.c.race_id == e.c.race_id)
                            .where(e.c.race_id.in_([int(x) for x in race_ids])), conn)
        if today.empty:
            return set()
        today = _latest(today)
        today = today[today["discipline"].astype(str).str.startswith("trot")
                      & (today["status"].fillna("").str.upper() != "NON_PARTANT")
                      & (today["shoes"] == D4)]
        if today.empty:
            return set()
        horses = [int(h) for h in today["horse_id"].unique()]
        hist = pd.read_sql(select(*cols, r.c.race_date).join(r, r.c.race_id == e.c.race_id)
                           .where(e.c.horse_id.in_(horses), r.c.race_date < day), conn)
    if hist.empty:
        return set()
    hist = _latest(hist)
    hist = hist[hist["status"].fillna("").str.upper() != "NON_PARTANT"]
    last = hist.sort_values(["race_date", "race_id"]).groupby("horse_id").tail(1)
    prev = dict(zip(last["horse_id"].astype(int),
                    last["shoes"].astype(object).where(last["shoes"].notna(), "FERRE"),
                    strict=True))
    return {(int(rid), int(num)) for rid, num, h in
            zip(today["race_id"], today["num_pmu"], today["horse_id"], strict=True)
            if int(h) in prev and prev[int(h)] != D4}
