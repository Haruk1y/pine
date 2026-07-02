from collections.abc import Iterable, Sequence
from functools import partial

import gurobipy as gp
import numpy.typing as npt

from ..ensemble import EnsembleContainer
from ..feature import FeatureContainer, FeatureEncoder, FeatureVars
from ..flow import FlowVars
from ..mip import MIP
from ..tree import Tree
from ..typing import BaseEnsemble, MNumber
from .constraints.strategy import ConstraintContext, ConstraintStrategy
from .parsers import LevelParser


class BaseOCEAN(
    MIP,
    EnsembleContainer,
    FeatureContainer,
):
    DEFAULT_TOL = 1e-4
    FEATURE_VARS_NAME = "feature_vars"
    FLOW_VAR_FMT = "tree_{t}"

    _feature_vars: FeatureVars
    _flow_vars: dict[int, FlowVars]
    _constraints: tuple[ConstraintStrategy, ...]

    _level_parser: LevelParser
    _levels: dict[str, MNumber]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
        *,
        constraints: Sequence[ConstraintStrategy] | None = None,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        tol: float = DEFAULT_TOL,
    ) -> None:
        MIP.__init__(self, name=name, env=env)
        EnsembleContainer.__init__(
            self,
            ensemble=(base, encoder),
            weights=weights,
        )
        FeatureContainer.__init__(self, encoder=encoder)
        self._init_constraints(constraints=constraints, encoder=encoder)
        self._parse_levels(encoder=encoder, tol=tol)
        self._add_feature_vars()
        self._add_flow_vars()

    @property
    def levels(self) -> dict[str, MNumber]:
        return self._levels

    def build(self) -> None:
        self._build_feature_vars()
        self._build_flow_vars()
        self._build_feature_constrs()
        self._build_constraint_vars()
        self._add_constraint_constrs()

    def function(self, class_: int) -> gp.LinExpr:
        weights = self._weights
        wf = partial(self.weighted_function, weights=weights)
        return wf(class_=class_)

    def weighted_function(self, class_: int, weights: MNumber) -> gp.LinExpr:
        return gp.quicksum(
            weights[t] * self._flow_function(t=t, class_=class_)
            for t in range(self.n_estimators)
        )

    def _flow_function(self, t: int, class_: int) -> gp.MLinExpr:
        if self._flow_vars[t].value.ndim == 0:
            n_classes = self.n_classes
            if self.is_binary:
                return (2 * class_ - 1) * self._flow_vars[t].value
            return self._flow_vars[t * n_classes + class_].value
        return self._flow_vars[t].value[class_]

    def _init_constraints(
        self,
        *,
        constraints: Sequence[ConstraintStrategy] | None,
        encoder: FeatureEncoder,
    ) -> None:
        self._constraints = tuple(constraints) if constraints else ()
        if not self._constraints:
            return
        base_ensemble = tuple(self.ensemble)
        for constraint in self._constraints:
            constraint.prepare(encoder=encoder, ensemble=base_ensemble)

    def _build_feature_vars(self) -> None:
        self._feature_vars.build(mip=self)

    def _build_flow_vars(self) -> None:
        for flow_vars in self._flow_vars.values():
            flow_vars.build(mip=self)

    def _build_feature_constrs(self) -> None:
        for flow_vars in self._flow_vars.values():
            flow_vars.add_feature_vars(
                mip=self,
                feature_vars=self._feature_vars,
            )

    def _parse_levels(
        self,
        encoder: FeatureEncoder,
        tol: float,
    ) -> None:
        self._level_parser = LevelParser(tol=tol)
        ensembles: list[Iterable[Tree]] = [self.ensemble]
        for constraint in self._constraints:
            extra = constraint.extra_trees()
            if extra:
                ensembles.append(extra)
        self._levels = self._level_parser.parse_levels(*ensembles, encoder=encoder)

    def _add_feature_vars(self) -> None:
        self._feature_vars = FeatureVars(name=self.FEATURE_VARS_NAME)
        for feature in self.features:
            self._add_feature_var(feature)

    def _add_feature_var(self, feature: str) -> None:
        vtype = self.types[feature].value
        levels = self.levels.get(feature)
        categories = self.categories.get(feature)
        self._feature_vars.add_var(
            feature=feature,
            vtype=vtype,
            levels=levels,
            categories=categories,
        )

    def _add_flow_vars(self) -> None:
        self._flow_vars = {}
        for t, tree in enumerate(self.ensemble):
            name = self.FLOW_VAR_FMT.format(t=t)
            self._flow_vars[t] = FlowVars(tree=tree, name=name)

    def _constraint_context(self) -> ConstraintContext:
        return ConstraintContext(
            mip=self,
            encoder=self.encoder,
            ensemble=tuple(self.ensemble),
            feature_vars=self._feature_vars,
            flow_vars=self._flow_vars,
        )

    def _build_constraint_vars(self) -> None:
        if not self._constraints:
            return
        ctx = self._constraint_context()
        for constraint in self._constraints:
            constraint.build_vars(ctx)

    def _add_constraint_constrs(self) -> None:
        if not self._constraints:
            return
        ctx = self._constraint_context()
        for constraint in self._constraints:
            constraint.add_constraints(ctx)

    def set_constraint_context(
        self,
        *,
        majority_class: int | None = None,
        target_class: int | None = None,
    ) -> None:
        if not self._constraints:
            return
        for constraint in self._constraints:
            constraint.set_class_context(
                majority_class=majority_class,
                target_class=target_class,
            )
        self.update()
