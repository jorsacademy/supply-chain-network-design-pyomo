from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .bounds import ComplementarityBounds, derive_complementarity_bounds
from .follower import follower_feasible, follower_objective, solve_follower
from .model import BilevelSolution, SupplyChainInstance


@dataclass(frozen=True)
class VariableSlices:
    w: slice
    q: slice
    lam: int
    mu: slice
    nu: slice
    z_capacity: int
    z_upper: slice
    z_lower: slice
    n_variables: int


def _layout(n: int) -> VariableSlices:
    start = 0
    w = slice(start, start + n)
    start += n
    q = slice(start, start + n)
    start += n
    lam = start
    start += 1
    mu = slice(start, start + n)
    start += n
    nu = slice(start, start + n)
    start += n
    z_capacity = start
    start += 1
    z_upper = slice(start, start + n)
    start += n
    z_lower = slice(start, start + n)
    start += n
    return VariableSlices(w, q, lam, mu, nu, z_capacity, z_upper, z_lower, start)


def _build_constraints(
    instance: SupplyChainInstance,
    bounds: ComplementarityBounds,
    layout: VariableSlices,
) -> LinearConstraint:
    n = instance.n_products
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for j in range(n):
        row = np.zeros(layout.n_variables)
        row[layout.w.start + j] = 1.0
        row[layout.lam] = instance.resource_use[j]
        row[layout.mu.start + j] = 1.0
        row[layout.nu.start + j] = -1.0
        rows.append(row)
        lower.append(float(instance.retail_price[j]))
        upper.append(float(instance.retail_price[j]))

    row = np.zeros(layout.n_variables)
    row[layout.q] = instance.resource_use
    rows.append(row)
    lower.append(-np.inf)
    upper.append(float(instance.retailer_capacity))

    row = np.zeros(layout.n_variables)
    row[layout.lam] = 1.0
    row[layout.z_capacity] = -bounds.capacity_dual
    rows.append(row)
    lower.append(-np.inf)
    upper.append(0.0)

    row = np.zeros(layout.n_variables)
    row[layout.q] = -instance.resource_use
    row[layout.z_capacity] = bounds.capacity_slack
    rows.append(row)
    lower.append(-np.inf)
    upper.append(0.0)

    for j in range(n):
        row = np.zeros(layout.n_variables)
        row[layout.mu.start + j] = 1.0
        row[layout.z_upper.start + j] = -bounds.upper_dual[j]
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(layout.n_variables)
        row[layout.q.start + j] = -1.0
        row[layout.z_upper.start + j] = bounds.upper_slack[j]
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(layout.n_variables)
        row[layout.nu.start + j] = 1.0
        row[layout.z_lower.start + j] = -bounds.lower_dual[j]
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(layout.n_variables)
        row[layout.q.start + j] = 1.0
        row[layout.z_lower.start + j] = bounds.lower_slack[j]
        rows.append(row)
        lower.append(-np.inf)
        upper.append(float(instance.quantity_cap[j]))

    return LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper))


def solve_kkt_milp(instance: SupplyChainInstance) -> BilevelSolution:
    """Solve the optimistic bilevel model through a KKT/Fortuny-Amat MILP."""

    n = instance.n_products
    layout = _layout(n)
    comp = derive_complementarity_bounds(instance)

    objective = np.zeros(layout.n_variables)
    objective[layout.q] = -(instance.retail_price - instance.production_cost)
    objective[layout.lam] = instance.retailer_capacity
    objective[layout.mu] = instance.quantity_cap

    lower = np.zeros(layout.n_variables)
    upper = np.full(layout.n_variables, np.inf)
    lower[layout.w] = instance.wholesale_lower
    upper[layout.w] = instance.wholesale_upper
    upper[layout.q] = instance.quantity_cap
    upper[layout.lam] = comp.capacity_dual
    upper[layout.mu] = comp.upper_dual
    upper[layout.nu] = comp.lower_dual
    upper[layout.z_capacity] = 1.0
    upper[layout.z_upper] = 1.0
    upper[layout.z_lower] = 1.0

    integrality = np.zeros(layout.n_variables, dtype=int)
    integrality[layout.z_capacity] = 1
    integrality[layout.z_upper] = 1
    integrality[layout.z_lower] = 1

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=_build_constraints(instance, comp, layout),
        options={"presolve": True},
    )
    if not result.success or result.x is None or result.fun is None:
        raise RuntimeError(f"KKT MILP failed: {result.message}")

    x = np.asarray(result.x, dtype=float)
    w = x[layout.w]
    q = x[layout.q]
    lam = float(x[layout.lam])
    mu = x[layout.mu]
    nu = x[layout.nu]

    manufacturer_profit_direct = float(np.dot(w - instance.production_cost, q))
    manufacturer_profit_dual = float(
        np.dot(instance.retail_price - instance.production_cost, q)
        - instance.retailer_capacity * lam
        - np.dot(instance.quantity_cap, mu)
    )
    if not np.isclose(manufacturer_profit_direct, manufacturer_profit_dual, atol=1e-6):
        raise RuntimeError("strong-duality profit identity failed")
    if not follower_feasible(instance, q, tol=1e-7):
        raise RuntimeError("KKT solution is not follower-primal feasible")
    follower = solve_follower(instance, w)
    q_value = follower_objective(instance, w, q)
    if not np.isclose(q_value, follower.retailer_profit, atol=1e-6):
        raise RuntimeError("KKT quantity is not follower-optimal")

    return BilevelSolution(
        wholesale_price=w,
        quantity=q,
        manufacturer_profit=manufacturer_profit_direct,
        retailer_profit=q_value,
        capacity_dual=lam,
        upper_dual=mu,
        lower_dual=nu,
        status="optimal",
    )
