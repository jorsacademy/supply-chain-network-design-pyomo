from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

from .follower import follower_feasible, follower_objective
from .model import SupplyChainInstance


@dataclass(frozen=True)
class ExactBilevelSolution:
    wholesale_price: np.ndarray
    quantity: np.ndarray
    manufacturer_profit: float
    retailer_profit: float
    extreme_points_checked: int
    leader_lps_solved: int


def enumerate_follower_extreme_points(
    instance: SupplyChainInstance,
    tol: float = 1e-9,
) -> list[np.ndarray]:
    """Enumerate extreme points of the tiny follower polytope."""

    n = instance.n_products
    active_rows: list[tuple[np.ndarray, float]] = []
    active_rows.append((instance.resource_use.copy(), float(instance.retailer_capacity)))
    for j in range(n):
        row = np.zeros(n)
        row[j] = 1.0
        active_rows.append((row, 0.0))
    for j in range(n):
        row = np.zeros(n)
        row[j] = 1.0
        active_rows.append((row, float(instance.quantity_cap[j])))

    points: list[np.ndarray] = []
    for selected in combinations(range(len(active_rows)), n):
        A_eq = np.stack([active_rows[index][0] for index in selected])
        if np.linalg.matrix_rank(A_eq, tol=tol) < n:
            continue
        b_eq = np.asarray([active_rows[index][1] for index in selected])
        try:
            q = np.linalg.solve(A_eq, b_eq)
        except np.linalg.LinAlgError:
            continue
        if not follower_feasible(instance, q, tol=1e-7):
            continue
        if not any(np.allclose(q, existing, atol=1e-8) for existing in points):
            points.append(q)
    if not points:
        raise RuntimeError("failed to enumerate follower extreme points")
    return points


def solve_exact_optimistic(instance: SupplyChainInstance) -> ExactBilevelSolution:
    """Solve the tiny optimistic bilevel problem by follower-vertex enumeration."""

    points = enumerate_follower_extreme_points(instance)
    best: ExactBilevelSolution | None = None
    lps = 0

    for qk in points:
        A_ub: list[np.ndarray] = []
        b_ub: list[float] = []
        for ql in points:
            diff = qk - ql
            A_ub.append(diff)
            b_ub.append(float(np.dot(instance.retail_price, diff)))

        result = linprog(
            c=-qk,
            A_ub=np.asarray(A_ub),
            b_ub=np.asarray(b_ub),
            bounds=list(zip(instance.wholesale_lower, instance.wholesale_upper, strict=True)),
            method="highs",
        )
        lps += 1
        if not result.success or result.x is None:
            continue
        w = np.asarray(result.x, dtype=float)
        manufacturer_profit = float(np.dot(w - instance.production_cost, qk))
        retailer_profit = follower_objective(instance, w, qk)
        candidate = ExactBilevelSolution(
            wholesale_price=w,
            quantity=qk.copy(),
            manufacturer_profit=manufacturer_profit,
            retailer_profit=retailer_profit,
            extreme_points_checked=len(points),
            leader_lps_solved=lps,
        )
        if best is None or candidate.manufacturer_profit > best.manufacturer_profit + 1e-8:
            best = candidate

    if best is None:
        raise RuntimeError("no follower extreme point admits a feasible leader price")
    return ExactBilevelSolution(
        wholesale_price=best.wholesale_price,
        quantity=best.quantity,
        manufacturer_profit=best.manufacturer_profit,
        retailer_profit=best.retailer_profit,
        extreme_points_checked=len(points),
        leader_lps_solved=lps,
    )
