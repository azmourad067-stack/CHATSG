"""Les migrations SQL versionnées doivent couvrir toutes les tables du modèle."""

from __future__ import annotations

import re
from pathlib import Path

from horse_intelligence.data.storage.database import metadata

MIGRATIONS = Path(__file__).resolve().parents[1] / "supabase" / "migrations"


def _all_sql() -> str:
    return "\n".join(p.read_text() for p in sorted(MIGRATIONS.glob("*.sql")))


def test_schema_migration_creates_every_table() -> None:
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _all_sql()))
    assert created == {t.name for t in metadata.sorted_tables}


def test_security_migration_protects_every_table() -> None:
    # chaque migration protège ses tables : RLS, rien pour l'API publique, lecture hi_app
    protected = {name for block in re.findall(r"foreach t in array array\[(.*?)\]", _all_sql(),
                                              flags=re.S)
                 for name in re.findall(r"'(\w+)'", block)}
    assert protected == {t.name for t in metadata.sorted_tables}
