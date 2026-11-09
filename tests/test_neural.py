"""Expérience deep learning : le réseau respecte le protocole sans fuite (torch requis)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from tests.synthetic import START_DAY, populate

from horse_intelligence.data.storage.repositories import Repository
from horse_intelligence.evaluation.metrics import compute_metrics
from horse_intelligence.intelligence.feature_engineering import (
    CATEGORICAL,
    build_dataset,
    load_raw,
)
from horse_intelligence.models.baseline import uniform_proba
from horse_intelligence.services.training import NUMERIC_FEATURES

pytest.importorskip("torch")

from horse_intelligence.models.neural_model import NeuralRaceModel
from horse_intelligence.services.experiment import neural_runners


@pytest.fixture
def frame(repo: Repository):  # type: ignore[no-untyped-def]
    populate(repo, days=30)
    f = build_dataset(load_raw(repo.engine)).frame
    return f[f["won"].notna()]


def _split(frame):  # type: ignore[no-untyped-def]
    d = frame["race_date"]
    base = frame[d < START_DAY + timedelta(days=18)]
    calib = frame[(d >= START_DAY + timedelta(days=18)) & (d < START_DAY + timedelta(days=24))]
    test = frame[d >= START_DAY + timedelta(days=24)]
    return base, calib, test


def test_probabilities_per_race_and_signal(frame) -> None:  # type: ignore[no-untyped-def]
    base, calib, test = _split(frame)
    m = NeuralRaceModel(NUMERIC_FEATURES, CATEGORICAL, with_market=True, max_epochs=15)
    p = m.fit(base, calib).predict(test)
    sums = test.assign(p=p).groupby("race_id")["p"].sum()
    assert np.allclose(sums, 1.0) and m.epochs_run >= 1
    # les capacités latentes du générateur synthétique sont apprises : mieux que le hasard
    assert compute_metrics(test, p).log_loss < compute_metrics(test, uniform_proba(test)).log_loss


def test_unknown_entities_do_not_break_prediction(frame) -> None:  # type: ignore[no-untyped-def]
    base, calib, test = _split(frame)
    m = NeuralRaceModel(NUMERIC_FEATURES, CATEGORICAL, max_epochs=2).fit(base, calib)
    odd = test.copy()
    odd["horse_id"] = 10_000_000          # chevaux jamais vus : profil « inconnu »
    odd["jockey_id"] = np.nan
    assert np.isfinite(m.predict(odd)).all()


def test_network_never_sees_test_period(frame) -> None:  # type: ignore[no-untyped-def]
    """Le réseau réutilisé d'un pli à l'autre n'a été appris que sur des dates antérieures."""
    fits: list[dict[str, object]] = []
    runners = neural_runners(refit_every_days=100, fits=fits)
    base, calib, test = _split(frame)
    runners["neural_market"](base, calib, test)
    later_base = frame[frame["race_date"] < START_DAY + timedelta(days=24)]
    runners["neural_market"](later_base, calib, test)   # réutilisé, pas réappris
    assert len(fits) == 1
    assert fits[0]["trained_until"] == str(base["race_date"].max())
    assert base["race_date"].max() < test["race_date"].min()
