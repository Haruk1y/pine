from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .prune import BasePruner
from .typing import MNumber


def evaluate(
    pruner: BasePruner,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    original_weights: MNumber,
) -> dict[str, float]:
    y_arr = np.asarray(y)
    pred = pruner.ensemble.predict(X, original_weights)
    new_pred = pruner.predict(X)
    accuracy = float((pred == y_arr).mean())
    pruner_accuracy = float((new_pred == y_arr).mean())
    fidelity = float((pred == new_pred).mean())
    return {
        "accuracy": pruner_accuracy,
        "accuracy.before_pruning": accuracy,
        "accuracy.after_pruning": pruner_accuracy,
        "fidelity": fidelity,
    }


def evaluate_on_mask(
    pruner: BasePruner,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    original_weights: MNumber,
    mask: npt.ArrayLike | None,
) -> dict[str, float]:
    if mask is None:
        return evaluate(pruner, X, y, original_weights)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    y_arr = np.asarray(y)
    if mask_arr.shape[0] != y_arr.shape[0]:
        msg = "mask must match the number of samples."
        raise ValueError(msg)
    if mask_arr.sum() == 0:
        return {
            "accuracy": float("nan"),
            "accuracy.before_pruning": float("nan"),
            "accuracy.after_pruning": float("nan"),
            "fidelity": float("nan"),
        }
    return evaluate(pruner, np.asarray(X)[mask_arr], y_arr[mask_arr], original_weights)


def pruning_rate(pruner: BasePruner) -> float:
    return 1.0 - (pruner.n_active_estimators / pruner.n_estimators)
