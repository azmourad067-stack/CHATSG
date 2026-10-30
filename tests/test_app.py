"""Tests de fumée de l'interface Streamlit (chaque page, base vide puis remplie)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.conftest import RACE_DAY, START
from tests.test_no_data_leakage import FakePmu

from horse_intelligence.config import settings as settings_module
from horse_intelligence.services.context import build_context
from horse_intelligence.services.ingestion import ingest_day

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["dashboard", "races", "horse_profile", "settings"]


def _page_script(module: str) -> None:
    import importlib

    from horse_intelligence.frontend.components.cards import inject_style

    inject_style()
    importlib.import_module(f"horse_intelligence.frontend.{module}").render()


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")
    _clear_caches()
    yield tmp_path
    _clear_caches()


def _clear_caches() -> None:
    import streamlit as st

    settings_module.get_settings.cache_clear()
    st.cache_resource.clear()
    st.cache_data.clear()


def _run_page(module: str) -> AppTest:
    at = AppTest.from_function(_page_script, args=(module,), default_timeout=30)
    at.run()
    return at


def test_app_entrypoint_starts(isolated_env) -> None:  # type: ignore[no-untyped-def]
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Tableau de bord"


@pytest.mark.parametrize("module", PAGES)
def test_pages_on_empty_database(isolated_env, module) -> None:  # type: ignore[no-untyped-def]
    assert not _run_page(module).exception


@pytest.mark.parametrize("module", PAGES)
def test_pages_with_data(isolated_env, module) -> None:  # type: ignore[no-untyped-def]
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), build_context().repo)
    assert not _run_page(module).exception


def test_race_detail_and_horse_search(isolated_env) -> None:  # type: ignore[no-untyped-def]
    ingest_day(RACE_DAY, FakePmu(START + timedelta(hours=2), True), build_context().repo)
    at = _run_page("races")
    assert not at.exception
    assert any("PRIX DE TEST" in m.value for m in at.markdown)

    at = _run_page("horse_profile")
    at.text_input[0].input("bolide").run()
    assert not at.exception
    assert at.dataframe  # historique affiché


def test_unavailable_pages_show_no_numbers(isolated_env) -> None:  # type: ignore[no-untyped-def]
    def script() -> None:
        from horse_intelligence.frontend.unavailable import make

        make("Analyse approfondie")()

    at = AppTest.from_function(script)
    at.run()
    assert not at.exception and not at.dataframe and not at.metric
