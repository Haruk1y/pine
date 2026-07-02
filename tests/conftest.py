from __future__ import annotations

import pytest


def make_gurobi_env():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is not available: {exc}")
    return env
