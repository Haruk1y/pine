from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...feature import FeatureEncoder, FeatureVars
from ...flow import FlowVars
from ...mip import MIP
from ...tree import Tree


@dataclass(frozen=True)
class ConstraintContext:
    mip: MIP
    encoder: FeatureEncoder
    ensemble: tuple[Tree, ...]
    feature_vars: FeatureVars
    flow_vars: dict[int, FlowVars]


@runtime_checkable
class ConstraintStrategy(Protocol):
    name: str

    def prepare(
        self,
        *,
        encoder: FeatureEncoder,
        ensemble: tuple[Tree, ...],
    ) -> None:
        ...

    def extra_trees(self) -> tuple[Tree, ...]:
        ...

    def build_vars(self, ctx: ConstraintContext) -> None:
        ...

    def add_constraints(self, ctx: ConstraintContext) -> None:
        ...

    def set_class_context(
        self,
        *,
        majority_class: int | None,
        target_class: int | None,
    ) -> None:
        ...


class BaseConstraintStrategy:
    name = "base"

    def prepare(
        self,
        *,
        encoder: FeatureEncoder,
        ensemble: tuple[Tree, ...],
    ) -> None:
        return None

    def extra_trees(self) -> tuple[Tree, ...]:
        return ()

    def build_vars(self, ctx: ConstraintContext) -> None:
        return None

    def add_constraints(self, ctx: ConstraintContext) -> None:
        return None

    def set_class_context(
        self,
        *,
        majority_class: int | None,
        target_class: int | None,
    ) -> None:
        return None


ConstraintFactory = Callable[..., ConstraintStrategy]
_CONSTRAINT_REGISTRY: dict[str, ConstraintFactory] = {}


def register_constraint(name: str) -> Callable[[ConstraintFactory], ConstraintFactory]:
    def deco(factory: ConstraintFactory) -> ConstraintFactory:
        _CONSTRAINT_REGISTRY[name] = factory
        return factory

    return deco


def create_constraint(name: str, **params: object) -> ConstraintStrategy:
    if name not in _CONSTRAINT_REGISTRY:
        available = ", ".join(sorted(_CONSTRAINT_REGISTRY))
        msg = f"Unknown constraint: {name}. available=[{available}]"
        raise ValueError(msg)
    return _CONSTRAINT_REGISTRY[name](**params)


def available_constraints() -> tuple[str, ...]:
    return tuple(sorted(_CONSTRAINT_REGISTRY))
