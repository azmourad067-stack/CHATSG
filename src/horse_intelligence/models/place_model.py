"""Probabilité de terminer dans les places payées (simple placé PMU).

Méthode actuelle de production : on déduit P(placé) des probabilités de victoire par la
formule de Harville (`ranking.place_probabilities`). Elle a un défaut connu : elle
surestime les chances de place des outsiders.

Modèle placé direct (`PlaceModel`) :
1. LightGBM (objectif binaire « finit dans les places payées ») appris sur la partie
   ancienne de l'apprentissage ;
2. régression logistique apprise sur la partie récente, qui combine ce score à une ou
   plusieurs probabilités de place « de Harville » (marché ou modèle gagnant) ;
3. ajustement par course : les probabilités sont décalées (en logit) pour que leur somme
   vaille le nombre de places payées, ce qui est vrai par construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from horse_intelligence.models.lightgbm_model import DEFAULT_PARAMS, NUM_ROUNDS, _logit
from horse_intelligence.models.ranking import FloatArray, paid_places, place_probabilities


def places_paid(frame: pd.DataFrame) -> pd.Series:
    """Places payées dans chaque course (selon les partants déclarés, à défaut les partants)."""
    declared = frame["declared_runners"].where(frame["declared_runners"].notna(),
                                               frame["n_runners"])
    return declared.map(lambda d: paid_places(int(d))).astype(float)


def placed_target(frame: pd.DataFrame) -> pd.Series:
    """1 si le cheval finit dans les places payées, 0 sinon (non classé compris),
    NaN sans résultat ou sans placé dans la course."""
    k = places_paid(frame)
    y = (frame["finish_position"].fillna(np.inf) <= k).astype(float)
    return y.where(frame["won"].notna() & (k > 0))


def harville_place(frame: pd.DataFrame, p_win: npt.ArrayLike) -> FloatArray:
    """P(placé) déduite des probabilités de victoire, course par course (Harville)."""
    p = pd.Series(np.asarray(p_win, dtype=np.float64), index=frame.index)
    out = pd.Series(np.nan, index=frame.index)
    k = places_paid(frame)
    for _, idx in frame.groupby("race_id").groups.items():
        g = p.loc[idx]
        n = int(k.loc[idx].iloc[0])
        if n and not g.isna().any():
            out.loc[idx] = place_probabilities(g.to_numpy(), n)
    return np.asarray(out.to_numpy(), dtype=np.float64)


def normalize_to_places(z: npt.ArrayLike, race_ids: npt.ArrayLike, k: npt.ArrayLike,
                        n_iter: int = 60) -> FloatArray:
    """Décale les logits `z` de chaque course pour que Σ sigmoïde = k (bissection).
    k est borné par le nombre de partants ; une course à k = 0 reçoit 0."""
    z = np.asarray(z, dtype=np.float64)
    codes, _ = pd.factorize(np.asarray(race_ids))
    n_races = codes.max() + 1 if codes.size else 0
    size = np.bincount(codes, minlength=n_races)
    target = np.minimum(np.bincount(codes, weights=np.asarray(k, dtype=np.float64),
                                    minlength=n_races) / np.maximum(size, 1), size)
    lo, hi = np.full(n_races, -30.0), np.full(n_races, 30.0)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        s = np.bincount(codes, weights=1 / (1 + np.exp(-(z + mid[codes]))), minlength=n_races)
        too_big = s > target
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    p = 1 / (1 + np.exp(-(z + ((lo + hi) / 2)[codes])))
    p[target[codes] <= 0] = 0.0
    return np.asarray(p, dtype=np.float64)


@dataclass
class PlaceModel:
    """LightGBM « placé » + logistique de combinaison + ajustement par course."""

    features: list[str]
    categorical: list[str]
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_rounds: int = NUM_ROUNDS
    booster: Any = None
    categories: dict[str, list[str]] = field(default_factory=dict)

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features].copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype("object"), categories=self.categories[c])
        return X

    def fit_score(self, base: pd.DataFrame) -> PlaceModel:
        import lightgbm as lgb

        b = base[placed_target(base).notna()]
        self.categories = {c: sorted(map(str, b[c].dropna().astype("object").unique()))
                           for c in self.categorical}
        data = lgb.Dataset(self._matrix(b), label=placed_target(b).to_numpy(),
                           categorical_feature=self.categorical, free_raw_data=False)
        self.booster = lgb.train(self.params, data, num_boost_round=self.num_rounds)
        return self

    def raw_score(self, frame: pd.DataFrame) -> FloatArray:
        return _logit(np.asarray(self.booster.predict(self._matrix(frame)), dtype=np.float64))


@dataclass
class PlaceCombiner:
    """Logistique sur des logits de place, puis ajustement par course (Σ p = places)."""

    coef: FloatArray | None = None
    intercept: float = 0.0

    def fit(self, inputs: FloatArray, calib: pd.DataFrame) -> PlaceCombiner:
        from sklearn.linear_model import LogisticRegression

        y = placed_target(calib).to_numpy()
        ok = ~np.isnan(y) & np.isfinite(inputs).all(axis=1)
        lr = LogisticRegression(C=1e3, max_iter=1000).fit(inputs[ok], y[ok].astype(int))
        self.coef, self.intercept = lr.coef_[0], float(lr.intercept_[0])
        return self

    def predict(self, inputs: FloatArray, frame: pd.DataFrame) -> FloatArray:
        if self.coef is None:
            raise RuntimeError("modèle non entraîné")
        z = inputs @ self.coef + self.intercept
        return normalize_to_places(z, frame["race_id"], places_paid(frame))


def place_log_losses(frame: pd.DataFrame, p: npt.ArrayLike) -> pd.Series:
    """Log-loss binaire « placé / non placé » sommée sur les partants, par course."""
    y = placed_target(frame).to_numpy()
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    return pd.Series(ll, index=frame.index).groupby(frame["race_id"]).sum()
