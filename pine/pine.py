from __future__ import annotations

from typing import Literal

import gurobipy as gp
import numpy as np
import numpy.typing as npt
import pandas as pd

from .ensemble import Ensemble
from .feature import FeatureEncoder
from .fipe import FIPE
from .ocean.constraints import ChowLiuStrategy
from .ocean.parsers import LevelParser
from .oracle import Oracle
from .plausibility import (
    ChowLiuModel,
    build_chow_liu_model,
    conformal_tau_chow_liu,
    score_chow_liu,
)
from .typing import BaseEnsemble, MClass, MNumber


class PINEPruner:
    """High-level PINE-CL wrapper around FIPE.

    Inputs to ``fit`` are encoded arrays in the same column order as
    ``FeatureEncoder.X``. A pandas DataFrame with ``encoder.columns`` is also
    accepted.
    """

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike | None = None,
        *,
        alpha: float = 0.2,
        n_bins: int = 4,
        beta: float = 1.0,
        norm: Literal[0, 1] = 1,
        max_oracle_calls: int = 100,
        name: str = "PINE",
        env: gp.Env | None = None,
        eps: float = Oracle.DEFAULT_EPS,
        tol: float = 1e-4,
    ) -> None:
        self.base = base
        self.encoder = encoder
        self.alpha = float(alpha)
        self.n_bins = int(n_bins)
        self.beta = float(beta)
        self.norm = norm
        self.max_oracle_calls = int(max_oracle_calls)
        self.name = name
        self.env = env
        self.eps = float(eps)
        self.tol = float(tol)

        self._input_weights = None if weights is None else np.asarray(weights, dtype=float)
        self.initial_weights_: MNumber | None = None
        self.chow_liu_model_: ChowLiuModel | None = None
        self.tau_: float | None = None
        self.strategy_: ChowLiuStrategy | None = None
        self.pruner_: FIPE | None = None
        self.levels_by_feature_: dict[str, np.ndarray] | None = None

    def fit(
        self,
        X_fit: npt.ArrayLike | pd.DataFrame,
        X_cal: npt.ArrayLike | pd.DataFrame,
        *,
        pruner_samples: npt.ArrayLike | pd.DataFrame | None = None,
        prune: bool = True,
    ) -> PINEPruner:
        X_fit_arr = self._as_encoded_array(X_fit)
        X_cal_arr = self._as_encoded_array(X_cal)
        if X_fit_arr.shape[0] == 0:
            msg = "X_fit must contain at least one sample."
            raise ValueError(msg)
        if X_cal_arr.shape[0] == 0:
            msg = "X_cal must contain at least one sample."
            raise ValueError(msg)

        ensemble = Ensemble(base=self.base, encoder=self.encoder)
        weights = self._resolve_weights(ensemble)
        self.initial_weights_ = weights.copy()
        self.levels_by_feature_ = LevelParser(tol=self.tol).parse_levels(
            ensemble,
            encoder=self.encoder,
        )
        self.chow_liu_model_ = build_chow_liu_model(
            X_fit_arr,
            encoder=self.encoder,
            n_bins=self.n_bins,
            beta=self.beta,
            levels_by_feature=self.levels_by_feature_,
        )
        self.tau_ = conformal_tau_chow_liu(
            self.chow_liu_model_,
            X_cal_arr,
            encoder=self.encoder,
            alpha=self.alpha,
        )
        self.strategy_ = ChowLiuStrategy(model=self.chow_liu_model_, tau=self.tau_)
        self.pruner_ = FIPE(
            self.base,
            self.encoder,
            weights,
            norm=self.norm,
            constraints=[self.strategy_],
            name=self.name,
            env=self.env,
            eps=self.eps,
            tol=self.tol,
            max_oracle_calls=self.max_oracle_calls,
        )
        self.pruner_.build()
        samples = X_fit_arr if pruner_samples is None else self._as_encoded_array(pruner_samples)
        self.pruner_.add_samples(samples)
        if prune:
            self.pruner_.prune()
        return self

    @property
    def weights(self) -> MNumber:
        return self._require_pruner().weights

    @property
    def active_estimators(self) -> set[int]:
        return self._require_pruner().active_estimators

    @property
    def n_active_estimators(self) -> int:
        return self._require_pruner().n_active_estimators

    @property
    def n_estimators(self) -> int:
        return self._require_pruner().n_estimators

    @property
    def n_oracle_calls(self) -> int:
        return self._require_pruner().n_oracle_calls

    @property
    def oracle_samples(self):
        return self._require_pruner().oracle_samples

    @property
    def ensemble(self):
        return self._require_pruner().ensemble

    def predict(self, X: npt.ArrayLike | pd.DataFrame) -> MClass:
        return self._require_pruner().predict(self._as_encoded_array(X))

    def predict_proba(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        return self._require_pruner().predict_proba(self._as_encoded_array(X))

    def score_chow_liu(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        model = self._require_model()
        return score_chow_liu(model, self._as_encoded_array(X), encoder=self.encoder)

    def in_region_mask(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        tau = self._require_tau()
        return self.score_chow_liu(X) <= tau

    @property
    def pruning_rate(self) -> float:
        pruner = self._require_pruner()
        return 1.0 - (pruner.n_active_estimators / pruner.n_estimators)

    def _resolve_weights(self, ensemble: Ensemble) -> MNumber:
        if self._input_weights is not None:
            weights = self._input_weights
        elif hasattr(self.base, "estimator_weights_"):
            weights = np.asarray(getattr(self.base, "estimator_weights_"), dtype=float)
        else:
            weights = np.ones(ensemble.n_estimators, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != ensemble.n_estimators:
            msg = (
                "weights must be a 1D array with length equal to the number "
                f"of estimators ({ensemble.n_estimators})."
            )
            raise ValueError(msg)
        return weights.astype(float, copy=True)

    def _as_encoded_array(self, X: npt.ArrayLike | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.encoder.columns if column not in X.columns]
            if missing:
                msg = (
                    "DataFrame inputs must already be encoded and include "
                    f"encoder columns. Missing: {missing[:5]}"
                )
                raise ValueError(msg)
            return X.loc[:, self.encoder.columns].to_numpy()
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            msg = "Expected a 2D encoded array."
            raise ValueError(msg)
        if X_arr.shape[1] != len(self.encoder.columns):
            msg = (
                "Encoded array has the wrong number of columns: "
                f"got {X_arr.shape[1]}, expected {len(self.encoder.columns)}."
            )
            raise ValueError(msg)
        return X_arr

    def _require_pruner(self) -> FIPE:
        if self.pruner_ is None:
            msg = "Call fit before using the fitted PINE pruner."
            raise RuntimeError(msg)
        return self.pruner_

    def _require_model(self) -> ChowLiuModel:
        if self.chow_liu_model_ is None:
            msg = "Call fit before using the Chow-Liu model."
            raise RuntimeError(msg)
        return self.chow_liu_model_

    def _require_tau(self) -> float:
        if self.tau_ is None:
            msg = "Call fit before using the conformal threshold."
            raise RuntimeError(msg)
        return self.tau_


PINE = PINEPruner
