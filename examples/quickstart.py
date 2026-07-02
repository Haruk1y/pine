from __future__ import annotations

import gurobipy as gp
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pine import FeatureEncoder, PINEPruner


def make_env() -> gp.Env | None:
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        return env
    except gp.GurobiError as exc:
        print(f"Gurobi is required for PINE pruning: {exc}")
        return None


def main() -> int:
    data = load_breast_cancer(as_frame=True)
    frame = pd.DataFrame(data.data)
    y = np.asarray(data.target)

    encoder = FeatureEncoder(frame)
    X = encoder.X.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y,
    )
    X_fit, X_cal, y_fit, _ = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=1,
        stratify=y_train,
    )

    model = XGBClassifier(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=1,
        random_state=0,
        verbosity=0,
    )
    model.fit(X_fit, y_fit)
    booster = model.get_booster()

    env = make_env()
    if env is None:
        return 0

    try:
        pine = PINEPruner(
            booster,
            encoder,
            alpha=0.2,
            n_bins=4,
            beta=1.0,
            norm=1,
            max_oracle_calls=10,
            env=env,
        ).fit(X_fit, X_cal)
    except gp.GurobiError as exc:
        print(f"Gurobi failed while solving the PINE MILPs: {exc}")
        return 0

    base_pred = pine.ensemble.predict(X_test, pine.initial_weights_)
    pruned_pred = pine.predict(X_test)
    id_mask = pine.in_region_mask(X_test)

    accuracy = float((pruned_pred == y_test).mean())
    fidelity = float((base_pred == pruned_pred).mean())
    coverage = float(id_mask.mean())
    id_fidelity = float((base_pred[id_mask] == pruned_pred[id_mask]).mean())

    print(f"accuracy={accuracy:.4f}")
    print(f"fidelity={fidelity:.4f}")
    print(f"in_region_coverage={coverage:.4f}")
    print(f"in_region_fidelity={id_fidelity:.4f}")
    print(f"pruning_rate={pine.pruning_rate:.4f}")
    print(f"active_estimators={pine.n_active_estimators}/{pine.n_estimators}")
    print(f"tau={pine.tau_:.6f}")
    print(f"oracle_calls={pine.n_oracle_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
