from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

import gurobipy as gp
import numpy as np

from ...feature import BinaryVar, CategoricalVar, ContinuousVar
from ...plausibility.chow_liu import ChowLiuModel
from ...typing import FeatureType
from .strategy import BaseConstraintStrategy, ConstraintContext, register_constraint


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


@dataclass
class _ChowLiuVars:
    bin_vars: dict[str, gp.MVar]
    state_exprs: dict[str, list[gp.LinExpr | gp.Var | gp.MVar]]
    pair_vars: dict[tuple[str, str], np.ndarray]


@register_constraint("chow_liu")
class ChowLiuStrategy(BaseConstraintStrategy):
    """MILP encoding of s_CL(x) <= tau for a fitted Chow-Liu model."""

    name = "chow_liu"

    def __init__(
        self,
        *,
        model: ChowLiuModel | None = None,
        tau: float | None = None,
        models_by_class: Mapping[int, ChowLiuModel] | None = None,
        taus_by_class: Mapping[int, float] | None = None,
        class_mode: str = "unconditional",
    ) -> None:
        if model is not None and models_by_class:
            msg = "Provide either model or models_by_class, not both."
            raise ValueError(msg)
        if model is None and not models_by_class:
            msg = "Chow-Liu constraint requires a model."
            raise ValueError(msg)
        self._model = model
        self._tau = float(tau) if tau is not None else None
        self._models_by_class = {
            int(key): value for key, value in (models_by_class or {}).items()
        }
        self._taus_by_class = {
            int(key): float(value) for key, value in (taus_by_class or {}).items()
        }
        self._class_mode = str(class_mode).lower()
        self._bin_vars: dict[str, gp.MVar] = {}
        self._state_exprs: dict[str, list[gp.LinExpr | gp.Var | gp.MVar]] = {}
        self._pair_vars: dict[tuple[str, str], np.ndarray] = {}
        self._model_vars_by_class: dict[int, _ChowLiuVars] = {}
        self._class_constraints: dict[int, gp.Constr] = {}
        self._active_classes: tuple[int, ...] = ()
        self._mip: gp.Model | None = None

    def build_vars(self, ctx: ConstraintContext) -> None:
        if self._models_by_class:
            if self._model_vars_by_class:
                return
            for class_id, model in self._models_by_class.items():
                prefix = self._name_prefix(class_id=class_id)
                self._model_vars_by_class[class_id] = self._build_model_vars(
                    ctx,
                    model,
                    name_prefix=prefix,
                )
            return

        if self._state_exprs or self._model is None:
            return
        vars_pack = self._build_model_vars(
            ctx,
            self._model,
            name_prefix=self._name_prefix(),
        )
        self._bin_vars = vars_pack.bin_vars
        self._state_exprs = vars_pack.state_exprs
        self._pair_vars = vars_pack.pair_vars

    def add_constraints(self, ctx: ConstraintContext) -> None:
        if self._models_by_class:
            self._mip = ctx.mip
            for class_id, model in self._models_by_class.items():
                vars_pack = self._model_vars_by_class.get(class_id)
                if vars_pack is None or not model.features:
                    continue
                expr = self._score_expr(model, vars_pack)
                constr = ctx.mip.addConstr(
                    expr <= gp.GRB.INFINITY,
                    name=f"chow_liu_c{class_id}",
                )
                self._class_constraints[class_id] = constr
            self._update_active_constraints()
            return

        if self._tau is None or self._model is None or not self._model.features:
            return

        vars_pack = _ChowLiuVars(
            bin_vars=self._bin_vars,
            state_exprs=self._state_exprs,
            pair_vars=self._pair_vars,
        )
        expr = self._score_expr(self._model, vars_pack)
        ctx.mip.addConstr(expr <= float(self._tau), name="chow_liu")

    def set_class_context(
        self,
        *,
        majority_class: int | None,
        target_class: int | None,
    ) -> None:
        if not self._models_by_class:
            return
        active = self._resolve_active_classes(
            majority_class=majority_class,
            target_class=target_class,
        )
        if active == self._active_classes:
            return
        self._active_classes = active
        self._update_active_constraints()

    def _resolve_active_classes(
        self,
        *,
        majority_class: int | None,
        target_class: int | None,
    ) -> tuple[int, ...]:
        mode = self._class_mode
        if mode not in {"target", "both"}:
            mode = "target"
        candidates: list[int] = []
        if mode == "both":
            if majority_class is not None:
                candidates.append(int(majority_class))
            if target_class is not None:
                candidates.append(int(target_class))
        elif target_class is not None:
            candidates.append(int(target_class))
        active = sorted({c for c in candidates if c in self._models_by_class})
        return tuple(active)

    def _update_active_constraints(self) -> None:
        if not self._class_constraints:
            return
        active = set(self._active_classes)
        for class_id, constr in self._class_constraints.items():
            tau = self._taus_by_class.get(class_id)
            if tau is None or class_id not in active:
                rhs = gp.GRB.INFINITY
            else:
                rhs = float(tau)
            constr.RHS = rhs
        if self._mip is not None:
            self._mip.update()

    @staticmethod
    def _name_prefix(*, class_id: int | None = None) -> str:
        if class_id is None:
            return "chow_liu_"
        return f"chow_liu_c{class_id}_"

    def _build_model_vars(
        self,
        ctx: ConstraintContext,
        model: ChowLiuModel,
        *,
        name_prefix: str,
    ) -> _ChowLiuVars:
        bin_vars: dict[str, gp.MVar] = {}
        state_exprs: dict[str, list[gp.LinExpr | gp.Var | gp.MVar]] = {}
        pair_vars: dict[tuple[str, str], np.ndarray] = {}

        for feature in model.features:
            ftype = model.types.get(feature)
            var = ctx.feature_vars.get(feature)
            if ftype == FeatureType.CON:
                if not isinstance(var, ContinuousVar):
                    continue
                bins = self._build_continuous_bins(
                    ctx,
                    feature,
                    var,
                    model=model,
                    name_prefix=name_prefix,
                )
                if bins is not None:
                    bin_vars[feature] = bins
            elif ftype == FeatureType.CAT:
                if not isinstance(var, CategoricalVar):
                    continue
            elif ftype == FeatureType.BIN:
                if not isinstance(var, BinaryVar):
                    continue

        for feature in model.features:
            ftype = model.types.get(feature)
            var = ctx.feature_vars.get(feature)
            if ftype == FeatureType.CON:
                bins = bin_vars.get(feature)
                if bins is None:
                    continue
                state_exprs[feature] = [bins[i] for i in range(bins.size)]
            elif ftype == FeatureType.CAT and isinstance(var, CategoricalVar):
                cats = model.categories.get(feature, ())
                state_exprs[feature] = [var[cat] for cat in cats]
            elif ftype == FeatureType.BIN and isinstance(var, BinaryVar):
                state_exprs[feature] = [1.0 - var.var, var.var]

        for child, parent in model.edge_scores:
            if child not in state_exprs or parent not in state_exprs:
                continue
            c_states = int(model.n_states[child])
            p_states = int(model.n_states[parent])
            pair = np.empty((c_states, p_states), dtype=object)
            for c_idx in range(c_states):
                for p_idx in range(p_states):
                    name = (
                        f"{name_prefix}pair_"
                        f"{_slug(child)}_{_slug(parent)}_"
                        f"{c_idx}_{p_idx}"
                    )
                    var = ctx.mip.addVar(vtype=gp.GRB.BINARY, name=name)
                    pair[c_idx, p_idx] = var
                    x = state_exprs[child][c_idx]
                    y = state_exprs[parent][p_idx]
                    ctx.mip.addConstr(var <= x, name=f"{name}_x")
                    ctx.mip.addConstr(var <= y, name=f"{name}_y")
                    ctx.mip.addConstr(var >= x + y - 1.0, name=f"{name}_xy")
            pair_vars[(child, parent)] = pair

        return _ChowLiuVars(
            bin_vars=bin_vars,
            state_exprs=state_exprs,
            pair_vars=pair_vars,
        )

    @staticmethod
    def _score_expr(model: ChowLiuModel, vars_pack: _ChowLiuVars) -> gp.LinExpr:
        expr = gp.LinExpr()
        root = model.root
        root_states = vars_pack.state_exprs.get(root)
        if root_states is None:
            msg = f"Missing root state expression for {root}."
            raise ValueError(msg)
        for idx, score in enumerate(model.root_scores):
            expr += float(score) * root_states[idx]

        for edge, scores in model.edge_scores.items():
            pair_vars = vars_pack.pair_vars.get(edge)
            if pair_vars is None:
                continue
            for c_idx in range(scores.shape[0]):
                for p_idx in range(scores.shape[1]):
                    expr += float(scores[c_idx, p_idx]) * pair_vars[c_idx, p_idx]
        return expr

    def _build_continuous_bins(
        self,
        ctx: ConstraintContext,
        feature: str,
        var: ContinuousVar,
        *,
        model: ChowLiuModel,
        name_prefix: str,
    ) -> gp.MVar | None:
        edges = model.bin_edges.get(feature)
        if edges is None:
            return None
        levels = np.asarray(var.levels, dtype=float)
        if levels.size == 0:
            return None
        n_bins = int(edges.size - 1)
        if n_bins <= 0:
            return None
        slug = _slug(feature)
        bins = ctx.mip.addMVar(
            shape=n_bins,
            vtype=gp.GRB.BINARY,
            name=f"{name_prefix}bin_{slug}",
        )
        ctx.mip.addConstr(
            bins.sum() == 1.0,
            name=f"{name_prefix}bin_sum_{slug}",
        )

        for idx in range(n_bins):
            lower = float(edges[idx])
            upper = float(edges[idx + 1])
            lower_expr = self._edge_indicator(levels, var, lower, is_lower=True)
            upper_expr = self._edge_indicator(levels, var, upper, is_lower=False)
            name = f"{name_prefix}bin_{slug}_{idx}"
            ctx.mip.addConstr(bins[idx] <= lower_expr, name=f"{name}_low")
            ctx.mip.addConstr(bins[idx] <= 1.0 - upper_expr, name=f"{name}_high")
            ctx.mip.addConstr(
                bins[idx] >= lower_expr - upper_expr,
                name=f"{name}_range",
            )
        return bins

    @staticmethod
    def _edge_indicator(
        levels: np.ndarray,
        var: ContinuousVar,
        edge: float,
        *,
        is_lower: bool,
    ) -> gp.LinExpr:
        if is_lower and math.isinf(edge) and edge < 0:
            return gp.LinExpr(1.0)
        if (not is_lower) and math.isinf(edge) and edge > 0:
            return gp.LinExpr(0.0)
        idx = int(np.searchsorted(levels, edge, side="left"))
        idx = min(max(idx, 0), max(levels.size - 1, 0))
        value = var[idx]
        if isinstance(value, gp.MVar):
            if value.size != 1:
                msg = f"Expected scalar MVar for edge indicator, got size={value.size}"
                raise ValueError(msg)
            value = value.item()
        return gp.LinExpr(value)
