from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from horse_intelligence.monitoring.logging_config import RedactingFilter, redact

ROOT = Path(__file__).resolve().parents[1]


def test_redact_connection_string_and_tokens() -> None:
    assert redact("postgresql://user:s3cr3t@db.host/x") == "postgresql://user:***@db.host/x"
    assert "abc123" not in redact("https://x.test/?api_key=abc123&a=1")
    assert "tok" not in redact("Authorization: Bearer tok")


def test_logging_filter_redacts(caplog) -> None:  # type: ignore[no-untyped-def]
    logger = logging.getLogger("hi.test")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO):
        logger.info("connexion %s", "postgresql://u:pw@h/db")
    assert "pw@" not in caplog.text


def test_secrets_are_gitignored() -> None:
    text = (ROOT / ".gitignore").read_text()
    for pattern in (".env", ".streamlit/secrets.toml", "data_store/", "*.db"):
        assert pattern in text


def test_no_hardcoded_secrets_in_tracked_files() -> None:
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,  # noqa: S607
                               check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")]
    suspicious = re.compile(
        r"(api[_-]?key|secret|password|token)\s*=\s*['\"][A-Za-z0-9_\-]{12,}['\"]", re.I)
    for name in files:
        path = ROOT / name
        if path.suffix in {".py", ".toml", ".yaml", ".yml"} and path.exists():
            assert not suspicious.search(path.read_text(errors="ignore")), name


def test_http_client_disables_redirects_and_sets_timeout() -> None:
    src = (ROOT / "src/horse_intelligence/data/collectors/base.py").read_text()
    assert "timeout=self.config.timeout_s" in src
    assert "allow_redirects=False" in src


def test_bad_database_url_is_explained_without_leaking() -> None:
    import pytest

    from horse_intelligence.data.storage.database import DatabaseUrlError, make_engine

    with pytest.raises(DatabaseUrlError, match="Session pooler") as exc:
        make_engine("https://projet.supabase.co?cle=secret123")
    assert "secret123" not in str(exc.value)
    # le format copié tel quel depuis Supabase (postgresql://) est accepté
    engine = make_engine("postgresql://u:p@localhost:5432/db")
    assert engine.dialect.driver == "psycopg"
