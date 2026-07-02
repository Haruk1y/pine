from __future__ import annotations

import numpy as np
import pandas as pd

from pine import (
    FeatureEncoder,
    build_chow_liu_model,
    conformal_tau_chow_liu,
    score_chow_liu,
)


def test_chow_liu_scoring_and_conformal_tau() -> None:
    frame = pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "b": [0, 1, 0, 1, 0, 1, 0, 1],
            "cat": pd.Series(["a", "b", "c", "a", "b", "c", "a", "b"], dtype="category"),
        }
    )
    encoder = FeatureEncoder(frame)
    X = encoder.X.to_numpy()

    model = build_chow_liu_model(X, encoder=encoder, n_bins=4, beta=1.0)
    scores = score_chow_liu(model, X, encoder=encoder)

    assert scores.shape == (8,)
    assert np.isfinite(scores).all()
    assert model.root in model.features
    assert set(model.parents).issubset(model.features)

    alpha = 0.25
    tau = conformal_tau_chow_liu(model, X, encoder=encoder, alpha=alpha)
    expected_idx = int(np.ceil((len(scores) + 1) * (1.0 - alpha))) - 1
    expected_idx = min(max(expected_idx, 0), len(scores) - 1)
    assert tau == np.sort(scores)[expected_idx]


def test_boundary_snapping_to_ensemble_levels() -> None:
    frame = pd.DataFrame({"x": np.arange(9, dtype=float), "b": [0, 1, 0] * 3})
    encoder = FeatureEncoder(frame)
    X = encoder.X.to_numpy()

    model = build_chow_liu_model(
        X,
        encoder=encoder,
        n_bins=4,
        beta=1.0,
        levels_by_feature={"x": np.array([1.0, 4.0, 8.0])},
    )

    assert np.allclose(model.bin_edges["x"], np.array([-np.inf, 1.0, 4.0, 8.0, np.inf]))
