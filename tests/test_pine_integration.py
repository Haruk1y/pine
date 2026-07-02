from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pine import FeatureEncoder, PINEPruner

from .conftest import make_gurobi_env


def test_small_pine_pruning_integration() -> None:
    env = make_gurobi_env()
    data = load_breast_cancer(as_frame=True)
    frame = pd.DataFrame(data.data).iloc[:80].copy()
    y = np.asarray(data.target[:80])
    encoder = FeatureEncoder(frame)
    X = encoder.X.to_numpy()

    X_fit, X_cal, y_fit, _ = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=0,
        stratify=y,
    )
    model = XGBClassifier(
        n_estimators=3,
        max_depth=1,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=1,
        random_state=0,
        verbosity=0,
    )
    model.fit(X_fit, y_fit)
    booster = model.get_booster()

    pine = PINEPruner(
        booster,
        encoder,
        alpha=0.2,
        n_bins=3,
        beta=1.0,
        max_oracle_calls=3,
        env=env,
    ).fit(X_fit, X_cal)

    assert pine.n_active_estimators > 0
    assert pine.weights.shape == (pine.n_estimators,)
    id_mask = pine.in_region_mask(X_cal)
    base_pred = pine.ensemble.predict(X_cal[id_mask], pine.initial_weights_)
    pruned_pred = pine.predict(X_cal[id_mask])
    assert np.all(base_pred == pruned_pred)
