from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pine import FeatureEncoder, PINEPruner


SKLEARN_DATASETS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
}


def load_dataset(args: argparse.Namespace) -> tuple[str, pd.DataFrame, np.ndarray]:
    if args.dataset_folder is not None:
        folder = Path(args.dataset_folder)
        name = folder.name
        full = folder / f"{name}.full.csv"
        if not full.exists():
            msg = f"Dataset folder must contain {full.name}."
            raise FileNotFoundError(msg)
        data = pd.read_csv(full, na_values=["?"]).dropna()
        label_col = data.columns[-1]
        y = data[label_col].astype("category").cat.codes.to_numpy()
        X = data.drop(columns=[label_col])
        return name, X, y

    loader = SKLEARN_DATASETS[args.dataset]
    data = loader(as_frame=True)
    return args.dataset, pd.DataFrame(data.data), np.asarray(data.target)


def make_model(args: argparse.Namespace):
    common = {"n_estimators": args.n_estimators, "random_state": args.seed}
    if args.model == "rf":
        return RandomForestClassifier(max_depth=args.max_depth, **common)
    if args.model == "gb":
        return GradientBoostingClassifier(max_depth=args.max_depth, **common)
    if args.model == "xgb":
        return XGBClassifier(
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            eval_metric="logloss",
            n_jobs=1,
            verbosity=0,
            **common,
        )
    if args.model == "ada":
        stump_depth = 1 if args.max_depth is None else args.max_depth
        estimator = None
        try:
            from sklearn.tree import DecisionTreeClassifier

            estimator = DecisionTreeClassifier(max_depth=stump_depth)
            return AdaBoostClassifier(estimator=estimator, **common)
        except TypeError:
            return AdaBoostClassifier(base_estimator=estimator, **common)
    msg = f"Unknown model: {args.model}"
    raise ValueError(msg)


def make_env(output_flag: int) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", output_flag)
    env.start()
    return env


def split_three_way(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float,
    calibration_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y if np.unique(y).size > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    stratify_train = y_train if np.unique(y_train).size > 1 else None
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=calibration_size,
        random_state=seed + 1,
        stratify=stratify_train,
    )
    return X_fit, X_cal, X_test, y_fit, y_cal, y_test


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_name, X_raw, y = load_dataset(args)
    encoder = FeatureEncoder(X_raw)
    X = encoder.X.to_numpy()
    if X.shape[0] != y.shape[0]:
        msg = "FeatureEncoder dropped rows; clean missing values before encoding."
        raise ValueError(msg)
    X_fit, X_cal, X_test, y_fit, _, y_test = split_three_way(
        X,
        y,
        seed=args.seed,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )

    model = make_model(args)
    model.fit(X_fit, y_fit)
    base = model.get_booster() if args.model == "xgb" else model
    env = make_env(args.gurobi_output)

    start = time.perf_counter()
    pine = PINEPruner(
        base,
        encoder,
        alpha=args.alpha,
        n_bins=args.n_bins,
        beta=args.beta,
        norm=args.norm,
        max_oracle_calls=args.max_oracle_calls,
        env=env,
    ).fit(X_fit, X_cal)
    elapsed = time.perf_counter() - start

    base_pred = pine.ensemble.predict(X_test, pine.initial_weights_)
    pruned_pred = pine.predict(X_test)
    id_mask = pine.in_region_mask(X_test)

    return {
        "dataset": dataset_name,
        "model": args.model,
        "seed": args.seed,
        "alpha": args.alpha,
        "n_bins": args.n_bins,
        "beta": args.beta,
        "n_estimators": pine.n_estimators,
        "n_active_estimators": pine.n_active_estimators,
        "pruning_rate": pine.pruning_rate,
        "accuracy": float((pruned_pred == y_test).mean()),
        "fidelity": float((base_pred == pruned_pred).mean()),
        "in_region_coverage": float(id_mask.mean()),
        "in_region_fidelity": float((base_pred[id_mask] == pruned_pred[id_mask]).mean()),
        "tau": pine.tau_,
        "oracle_calls": pine.n_oracle_calls,
        "runtime_sec": elapsed,
    }


def write_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PINE-CL with a three-way split.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        choices=sorted(SKLEARN_DATASETS),
        default="breast_cancer",
    )
    source.add_argument("--dataset-folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/pine_chowliu.csv"))
    parser.add_argument("--model", choices=["xgb", "gb", "rf", "ada"], default="xgb")
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--norm", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-oracle-calls", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--gurobi-output", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row = run(args)
    write_row(args.output, row)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
