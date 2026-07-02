# PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence

Official implementation of **PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence**.

PINE is a pruning method that guarantees prediction equivalence only within an in-distribution region, achieving a better trade-off between compression and prediction agreement.

For more details:
https://haruk1y.github.io/pine-icml/

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/Haruk1y/pine.git
cd pine
pip install -e .
```

The default installation includes the dependencies needed for the examples in this repository: `numpy`, `pandas`, `scikit-learn`, `xgboost`, and `gurobipy`.

If you want to use LightGBM models, install the optional booster extra:

```bash
pip install -e ".[boosters]"
```

## Gurobi

PINE requires the Gurobi solver for pruning. Free academic licenses are available. Please consult:

- [Gurobi academic program and licenses](https://www.gurobi.com/academia/academic-program-and-licenses/)
- [Gurobi academic license agreement](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)

### Getting started

A minimal working example is available at:

```bash
python examples/quickstart.py
```

The script trains a small XGBoost ensemble on a sklearn dataset, fits PINE-CL, and prints:

- `accuracy`: pruned model accuracy against labels.
- `fidelity`: prediction agreement between original and pruned ensemble on the test split.
- `in_region_coverage`: fraction of test points with `s_CL(x) <= tau`.
- `in_region_fidelity`: fidelity restricted to the conformal region.
- `pruning_rate`: fraction of estimators removed.

The same workflow can be written directly in Python:

```python
from pine import FeatureEncoder, PINEPruner
from xgboost import XGBClassifier

encoder = FeatureEncoder(raw_dataframe)
X = encoder.X.to_numpy()

xgb = XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.1)
xgb.fit(X_fit, y_fit)

pine = PINEPruner(
    xgb.get_booster(),
    encoder,
    alpha=0.2,
    n_bins=4,
    beta=1.0,
    max_oracle_calls=100,
).fit(X_fit, X_cal)

y_pruned = pine.predict(X_test)
id_mask = pine.in_region_mask(X_test)
```

If you want to use the FIPE loop directly, pass an explicit Chow-Liu constraint:

```python
from pine import (
    ChowLiuStrategy,
    FIPE,
    build_chow_liu_model,
    conformal_tau_chow_liu,
)

model = build_chow_liu_model(X_fit, encoder=encoder, n_bins=4, beta=1.0)
tau = conformal_tau_chow_liu(model, X_cal, encoder=encoder, alpha=0.2)

pruner = FIPE(
    trained_ensemble,
    encoder,
    weights,
    constraints=[ChowLiuStrategy(model=model, tau=tau)],
)
pruner.build()
pruner.add_samples(X_fit)
pruner.prune()
```

## CLI

Run a three-way split experiment on a sklearn built-in dataset:

```bash
python scripts/run_pine_chowliu.py \
  --dataset breast_cancer \
  --model xgb \
  --alpha 0.2 \
  --n-bins 4 \
  --output results/pine_chowliu.csv
```

Run on a dataset folder containing `<folder-name>.full.csv` with the label in the last column:

```bash
python scripts/run_pine_chowliu.py \
  --dataset-folder data/MyDataset \
  --output results/my_dataset.csv
```

Aggregate one or more CSV files:

```bash
python scripts/aggregate_results.py results/*.csv --output results/summary.csv
```

## Acknowledgements

Parts of the PINE implementation build on prior work from FIPE and OCEAN. We thank the authors for making these foundations available:

- FIPE: https://github.com/eminyous
- OCEAN: https://github.com/eminyous/ocean

## Citation

```bibtex
@inproceedings{yajima2026pine,
  title = {PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence},
  author = {Yajima, Haruki and Matsui, Yusuke},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year = {2026}
}
```
