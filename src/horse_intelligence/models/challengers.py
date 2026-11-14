"""Modèles candidats à LightGBM (expérience `models`) : même schéma que LgbmRaceModel.

1. un modèle de score appris sur la partie ancienne de l'apprentissage ;
2. un logit conditionnel par course, appris sur la partie récente, qui convertit le score
   en probabilités (et le combine à log(p_marché) dans la variante « + marché »).

Candidats :
- `LgbmRankScore` : LightGBM avec un objectif de classement PAR COURSE (rank_xendcg,
  entropie croisée par groupe) au lieu de l'objectif binaire cheval par cheval ;
- `CatBoostScore` : CatBoost (boosting ordonné, encodage natif des variables catégorielles) ;
- l'empilement de plusieurs scores, combinés par un logit conditionnel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.models.lightgbm_model import DEFAULT_PARAMS, NUM_ROUNDS, _logit
from horse_intelligence.models.ranking import ConditionalLogit, FloatArray

RANK_PARAMS: dict[str, Any] = {**DEFAULT_PARAMS, "objective": "rank_xendcg"}
CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 400, "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 5.0,
    "loss_function": "Logloss", "random_seed": 42, "thread_count": 4, "verbose": False,
    "allow_writing_files": False,
}
MISSING = "NA"


@dataclass
class LgbmRankScore:
    """LightGBM appris à classer les partants de chaque course (le vainqueur en tête)."""

    features: list[str]
    categorical: list[str]
    params: dict[str, Any] = field(default_factory=lambda: dict(RANK_PARAMS))
    num_rounds: int = NUM_ROUNDS
    booster: Any = None
    categories: dict[str, list[str]] = field(default_factory=dict)

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features].copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype("object"), categories=self.categories[c])
        return X

    def fit(self, base: pd.DataFrame) -> LgbmRankScore:
        import lightgbm as lgb

        b = base.sort_values("race_id", kind="stable")
        self.categories = {c: sorted(map(str, b[c].dropna().astype("object").unique()))
                           for c in self.categorical}
        groups = b.groupby("race_id", sort=False).size().to_numpy()
        data = lgb.Dataset(self._matrix(b), label=b["won"].astype(int).to_numpy(),
                           group=groups, categorical_feature=self.categorical,
                           free_raw_data=False)
        self.booster = lgb.train(self.params, data, num_boost_round=self.num_rounds)
        return self

    def raw_score(self, frame: pd.DataFrame) -> FloatArray:
        return np.asarray(self.booster.predict(self._matrix(frame)), dtype=np.float64)


@dataclass
class CatBoostScore:
    """CatBoost (objectif binaire « gagne ») ; score = logit brut du modèle."""

    features: list[str]
    categorical: list[str]
    params: dict[str, Any] = field(default_factory=lambda: dict(CATBOOST_PARAMS))
    model: Any = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features].copy()
        for c in self.categorical:
            X[c] = X[c].astype("object").where(X[c].notna(), MISSING).astype(str)
        return X

    def fit(self, base: pd.DataFrame) -> CatBoostScore:
        from catboost import CatBoostClassifier  # dépendance des expériences uniquement

        self.model = CatBoostClassifier(**self.params, cat_features=self.categorical)
        self.model.fit(self._matrix(base), base["won"].astype(int).to_numpy())
        return self

    def raw_score(self, frame: pd.DataFrame) -> FloatArray:
        return np.asarray(self.model.predict(self._matrix(frame),
                                             prediction_type="RawFormulaVal"), dtype=np.float64)


@dataclass
class CalibratedScores:
    """Logit conditionnel par course sur un ou plusieurs scores (+ log p_marché)."""

    scorers: dict[str, Any]
    with_market: bool = True
    calibrator: ConditionalLogit | None = None

    def _inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = {name: s.raw_score(frame) for name, s in self.scorers.items()}
        if self.with_market:
            cols["mkt_log_prob"] = frame["mkt_log_prob"].to_numpy()
        return pd.DataFrame(cols, index=frame.index)

    def fit(self, calib: pd.DataFrame) -> CalibratedScores:
        Z = self._inputs(calib)
        self.calibrator = ConditionalLogit(l2=1e-3).fit(Z, calib["race_id"], calib["won"],
                                                        list(Z.columns))
        return self

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        if self.calibrator is None:
            raise RuntimeError("modèle non entraîné")
        return self.calibrator.predict_proba(self._inputs(frame), frame["race_id"])


@dataclass
class LgbmBinaryScore:
    """Score du LightGBM de production (objectif binaire), réutilisable dans un empilement."""

    features: list[str]
    categorical: list[str]
    booster: Any = None
    categories: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, base: pd.DataFrame) -> LgbmBinaryScore:
        import lightgbm as lgb

        self.categories = {c: sorted(map(str, base[c].dropna().astype("object").unique()))
                           for c in self.categorical}
        data = lgb.Dataset(self._matrix(base), label=base["won"].to_numpy(),
                           categorical_feature=self.categorical, free_raw_data=False)
        self.booster = lgb.train(dict(DEFAULT_PARAMS), data, num_boost_round=NUM_ROUNDS)
        return self

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features].copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype("object"), categories=self.categories[c])
        return X

    def raw_score(self, frame: pd.DataFrame) -> FloatArray:
        return _logit(np.asarray(self.booster.predict(self._matrix(frame)), dtype=np.float64))
