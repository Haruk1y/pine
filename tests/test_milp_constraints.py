from __future__ import annotations

import numpy as np
import pandas as pd

from pine import FeatureEncoder, build_chow_liu_model
from pine.feature import FeatureVars
from pine.mip import MIP
from pine.ocean.constraints import ChowLiuStrategy
from pine.ocean.constraints.strategy import ConstraintContext

from .conftest import make_gurobi_env


def test_chow_liu_binary_milp_constraint_excludes_high_nll_state() -> None:
    env = make_gurobi_env()
    import gurobipy as gp

    frame = pd.DataFrame({"b": [0, 0, 0, 1]})
    encoder = FeatureEncoder(frame)
    X = encoder.X.to_numpy()
    model = build_chow_liu_model(X, encoder=encoder, beta=1.0)

    mip = MIP(name="test_chow_liu_binary", env=env)
    feature_vars = FeatureVars(name="feature_vars")
    feature_vars.add_var("b", encoder.types["b"].value)
    feature_vars.build(mip)

    ctx = ConstraintContext(
        mip=mip,
        encoder=encoder,
        ensemble=(),
        feature_vars=feature_vars,
        flow_vars={},
    )
    strategy = ChowLiuStrategy(model=model, tau=0.5)
    strategy.build_vars(ctx)
    strategy.add_constraints(ctx)

    mip.setObjective(feature_vars["b"].var, gp.GRB.MAXIMIZE)
    mip.optimize()

    assert mip.Status == gp.GRB.OPTIMAL
    assert np.isclose(feature_vars["b"].var.X, 0.0)
