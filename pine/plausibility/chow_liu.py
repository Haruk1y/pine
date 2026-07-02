from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..feature import FeatureEncoder
from ..typing import FeatureType


@dataclass(frozen=True)
class ChowLiuModel:
    """Discrete Chow-Liu tree represented as NLL lookup tables."""

    features: tuple[str, ...]
    types: dict[str, FeatureType]
    categories: dict[str, tuple[str, ...]]
    bin_edges: dict[str, np.ndarray]
    n_states: dict[str, int]
    root: str
    parents: dict[str, str | None]
    root_scores: np.ndarray
    edge_scores: dict[tuple[str, str], np.ndarray]


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        msg = "X must be a 2D array."
        raise ValueError(msg)
    return X


def _feature_order(encoder: FeatureEncoder) -> list[str]:
    return list(encoder.types.keys())


def _select_levels(
    levels_by_feature: Mapping[str, np.ndarray] | None,
    feature: str,
) -> np.ndarray | None:
    if levels_by_feature is None:
        return None
    levels = levels_by_feature.get(feature)
    if levels is None:
        return None
    levels = np.asarray(levels, dtype=float)
    if levels.size == 0:
        return None
    return levels


def _snap_to_levels(value: float, levels: np.ndarray) -> float:
    idx = int(np.searchsorted(levels, value, side="left"))
    if idx <= 0:
        return float(levels[0])
    if idx >= levels.size:
        return float(levels[-1])
    left = float(levels[idx - 1])
    right = float(levels[idx])
    return right if abs(right - value) <= abs(value - left) else left


def _make_bin_edges(
    values: np.ndarray,
    *,
    n_bins: int,
    levels: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or n_bins <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if levels is None or levels.size == 0:
        edges = np.quantile(values, quantiles)
        edges = sorted({float(edge) for edge in edges})
    else:
        targets = np.quantile(values, quantiles)
        edges = sorted({_snap_to_levels(float(edge), levels) for edge in targets})
    return np.array([-np.inf, *edges, np.inf], dtype=float)


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, edges.size - 2).astype(int)


def _discretize_for_fit(
    X: np.ndarray,
    *,
    encoder: FeatureEncoder,
    n_bins: int,
    levels_by_feature: Mapping[str, np.ndarray] | None,
) -> tuple[
    list[str],
    dict[str, FeatureType],
    dict[str, tuple[str, ...]],
    dict[str, np.ndarray],
    dict[str, int],
    np.ndarray,
]:
    X = _ensure_2d(X)
    column_index = {name: i for i, name in enumerate(encoder.columns)}
    features: list[str] = []
    types: dict[str, FeatureType] = {}
    categories: dict[str, tuple[str, ...]] = {}
    bin_edges: dict[str, np.ndarray] = {}
    n_states: dict[str, int] = {}
    discrete_columns: list[np.ndarray] = []

    for feature in _feature_order(encoder):
        ftype = encoder.types.get(feature)
        if ftype is None:
            continue
        if ftype == FeatureType.CAT:
            cats = tuple(sorted(encoder.categories.get(feature, ())))
            indices = [column_index[c] for c in cats if c in column_index]
            if len(indices) < 2:
                continue
            values = X[:, indices]
            state = np.argmax(values, axis=1).astype(int)
            n_states[feature] = len(indices)
            categories[feature] = cats
        elif ftype == FeatureType.BIN:
            idx = column_index.get(feature)
            if idx is None:
                continue
            values = np.asarray(X[:, idx], dtype=float)
            state = np.clip(np.round(values), 0, 1).astype(int)
            n_states[feature] = 2
        elif ftype == FeatureType.CON:
            idx = column_index.get(feature)
            if idx is None:
                continue
            values = X[:, idx]
            levels = _select_levels(levels_by_feature, feature)
            if levels_by_feature is not None and levels is None:
                continue
            edges = _make_bin_edges(values, n_bins=n_bins, levels=levels)
            if edges.size < 3:
                continue
            state = _digitize(values, edges)
            n_states[feature] = edges.size - 1
            bin_edges[feature] = edges
        else:
            continue
        if n_states.get(feature, 0) < 2:
            continue
        features.append(feature)
        types[feature] = ftype
        discrete_columns.append(state)

    if not features:
        msg = "Chow-Liu requires at least one discrete feature."
        raise ValueError(msg)

    discrete = np.stack(discrete_columns, axis=1)
    return features, types, categories, bin_edges, n_states, discrete


def _discretize_from_model(
    X: np.ndarray,
    *,
    encoder: FeatureEncoder,
    model: ChowLiuModel,
) -> np.ndarray:
    X = _ensure_2d(X)
    column_index = {name: i for i, name in enumerate(encoder.columns)}
    columns: list[np.ndarray] = []
    for feature in model.features:
        ftype = model.types.get(feature)
        if ftype == FeatureType.CAT:
            cats = model.categories.get(feature, ())
            indices = [column_index[c] for c in cats if c in column_index]
            if len(indices) != len(cats):
                msg = f"Missing categorical columns for {feature}."
                raise ValueError(msg)
            values = X[:, indices]
            state = np.argmax(values, axis=1).astype(int)
        elif ftype == FeatureType.BIN:
            idx = column_index.get(feature)
            if idx is None:
                msg = f"Missing binary column for {feature}."
                raise ValueError(msg)
            values = np.asarray(X[:, idx], dtype=float)
            state = np.clip(np.round(values), 0, 1).astype(int)
        elif ftype == FeatureType.CON:
            idx = column_index.get(feature)
            if idx is None:
                msg = f"Missing continuous column for {feature}."
                raise ValueError(msg)
            edges = model.bin_edges.get(feature)
            if edges is None:
                msg = f"Missing bin edges for {feature}."
                raise ValueError(msg)
            values = X[:, idx]
            state = _digitize(values, edges)
        else:
            msg = f"Unsupported feature type for {feature}."
            raise ValueError(msg)
        columns.append(state)
    return np.stack(columns, axis=1) if columns else np.zeros((len(X), 0), dtype=int)


def _joint_counts(
    x: np.ndarray,
    y: np.ndarray,
    n_x: int,
    n_y: int,
) -> np.ndarray:
    flat = np.asarray(x) * n_y + np.asarray(y)
    counts = np.bincount(flat, minlength=n_x * n_y)
    return counts.reshape(n_x, n_y)


def _mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_x: int,
    n_y: int,
) -> float:
    counts = _joint_counts(x, y, n_x, n_y).astype(float)
    n_total = counts.sum()
    if n_total <= 0:
        return 0.0
    pxy = counts / n_total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = px[:, None] * py[None, :]
        ratio = np.where(pxy > 0, pxy / denom, 1.0)
        mi = np.sum(np.where(pxy > 0, pxy * np.log(ratio), 0.0))
    return float(mi)


def _maximum_spanning_tree(weights: np.ndarray, *, root: int = 0) -> list[int]:
    n = weights.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, -np.inf, dtype=float)
    parent = np.full(n, -1, dtype=int)
    best[root] = 0.0
    for _ in range(n):
        candidates = np.where(~in_tree)[0]
        if candidates.size == 0:
            break
        u = candidates[np.argmax(best[candidates])]
        in_tree[u] = True
        for v in range(n):
            if in_tree[v]:
                continue
            w = weights[u, v]
            if w > best[v]:
                best[v] = w
                parent[v] = u
    return parent.tolist()


def build_chow_liu_model(
    X: np.ndarray,
    *,
    encoder: FeatureEncoder,
    n_bins: int = 4,
    beta: float = 1.0,
    levels_by_feature: Mapping[str, np.ndarray] | None = None,
) -> ChowLiuModel:
    """Fit a Laplace-smoothed Chow-Liu NLL model on encoded data."""

    X = _ensure_2d(X)
    if X.size == 0:
        msg = "Training data for Chow-Liu is empty."
        raise ValueError(msg)
    n_bins = int(n_bins)
    if n_bins < 1:
        msg = "n_bins must be >= 1."
        raise ValueError(msg)
    beta = float(beta)
    if beta <= 0.0:
        msg = "beta must be positive."
        raise ValueError(msg)

    (
        features,
        types,
        categories,
        bin_edges,
        n_states,
        discrete,
    ) = _discretize_for_fit(
        X,
        encoder=encoder,
        n_bins=n_bins,
        levels_by_feature=levels_by_feature,
    )
    n_features = len(features)
    if n_features == 0:
        msg = "Chow-Liu requires at least one discrete feature."
        raise ValueError(msg)

    counts = [
        np.bincount(discrete[:, i], minlength=n_states[features[i]]).astype(float)
        for i in range(n_features)
    ]

    if n_features == 1:
        root = features[0]
        denom = discrete.shape[0] + beta * n_states[root]
        root_probs = (counts[0] + beta) / denom
        return ChowLiuModel(
            features=tuple(features),
            types=types,
            categories=categories,
            bin_edges=bin_edges,
            n_states=n_states,
            root=root,
            parents={root: None},
            root_scores=-np.log(root_probs),
            edge_scores={},
        )

    mi = np.zeros((n_features, n_features), dtype=float)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi_ij = _mutual_information(
                discrete[:, i],
                discrete[:, j],
                n_states[features[i]],
                n_states[features[j]],
            )
            mi[i, j] = mi_ij
            mi[j, i] = mi_ij

    parents_idx = _maximum_spanning_tree(mi, root=0)
    root = features[0]
    parents: dict[str, str | None] = {root: None}
    edge_scores: dict[tuple[str, str], np.ndarray] = {}

    for child_idx, parent_idx in enumerate(parents_idx):
        if child_idx == 0:
            continue
        child = features[child_idx]
        parent = features[parent_idx]
        parents[child] = parent
        joint = _joint_counts(
            discrete[:, child_idx],
            discrete[:, parent_idx],
            n_states[child],
            n_states[parent],
        ).astype(float)
        denom = counts[parent_idx] + beta * n_states[child]
        cond = (joint + beta) / denom[None, :]
        edge_scores[(child, parent)] = -np.log(cond)

    denom = discrete.shape[0] + beta * n_states[root]
    root_probs = (counts[0] + beta) / denom
    root_scores = -np.log(root_probs)

    return ChowLiuModel(
        features=tuple(features),
        types=types,
        categories=categories,
        bin_edges=bin_edges,
        n_states=n_states,
        root=root,
        parents=parents,
        root_scores=root_scores,
        edge_scores=edge_scores,
    )


def score_chow_liu(
    model: ChowLiuModel,
    X: np.ndarray,
    *,
    encoder: FeatureEncoder,
) -> np.ndarray:
    """Return root plus edge conditional negative log likelihood scores."""

    X = _ensure_2d(X)
    if not model.features:
        return np.zeros(len(X), dtype=float)
    discrete = _discretize_from_model(X, encoder=encoder, model=model)
    feature_index = {name: i for i, name in enumerate(model.features)}
    root_idx = feature_index[model.root]
    values = model.root_scores[discrete[:, root_idx]].astype(float)
    for (child, parent), scores in model.edge_scores.items():
        c_idx = feature_index[child]
        p_idx = feature_index[parent]
        values += scores[discrete[:, c_idx], discrete[:, p_idx]]
    return values


def conformal_tau_chow_liu(
    model: ChowLiuModel,
    X_cal: np.ndarray,
    *,
    encoder: FeatureEncoder,
    alpha: float = 0.05,
) -> float:
    """Split conformal threshold using k=ceil((n+1)(1-alpha))."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        msg = "alpha must be in [0, 1]."
        raise ValueError(msg)
    values = score_chow_liu(model, X_cal, encoder=encoder)
    if values.size == 0:
        msg = "Calibration set is empty."
        raise ValueError(msg)
    values = np.sort(values)
    k = int(np.ceil((len(values) + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), len(values) - 1)
    return float(values[k])
