"""Les migrations SQL versionnées doivent couvrir toutes les tables du modèle."""

from __future__ import annotations

import re
from pathlib import Path

from horse_intelligence.data.storage.database import metadata

MIGRATIONS = Path(__file__).resolve().parents[1] / "supabase" / "migrations"


def test_schema_migration_creates_every_table() -> None:
    sql = (MIGRATIONS / "20261030120000_horse_intelligence_schema.sql").read_text()
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", sql))
    assert created == {t.name for t in metadata.sorted_tables}


def test_security_migration_protects_every_table() -> None:
    sql = (MIGRATIONS / "20261030120100_horse_intelligence_security.sql").read_text()
    protected = set(re.findall(r"'(\w+)'", sql.split("array[", 1)[1].split("]", 1)[0]))
    assert protected == {t.name for t in metadata.sorted_tables}
