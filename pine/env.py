from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    """Package-level runtime options.

    PINE's Pruner and Oracle formulations are Gurobi MILPs. The class is kept
    for compatibility with FIPE-style imports, but solver selection is no
    longer exposed in this public implementation.
    """

    solver: str = "gurobi"


ENV = Environment()
