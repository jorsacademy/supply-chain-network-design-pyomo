from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .model import FollowerSolution, SupplyChainInstance


def solve_follower(instance: SupplyChainInstance, wholesale_price: np.ndarray) -> FollowerSolution:
    """Solve the retailer's lower-level linear program exactly with HiGHS."""

    w = np.asarray(wholesale_price, dtype=float)
    if w.shape != instance.retail_price.shape:
        raise ValueError("wholesale_price has the wrong shape")
    result = linprog(
        c=w - instance.retail_price,
        A_ub=instance.resource_use.reshape(1, -1),
        b_ub=np.asarray([instance.retailer_capacity]),
        bounds=[(0.0, float(cap)) for cap in instance.quantity_cap],
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"follower LP failed: {result.message}")

    q = np.asarray(result.x, dtype=float)
    retailer_profit = float(np.dot(instance.retail_price - w, q))

    # SciPy reports marginals for the minimization form. For A q <= b, the
    # maximization-form nonnegative multiplier is the negative minimization marginal.
    capacity_dual = float(-result.ineqlin.marginals[0])
    lower_marginals = np.asarray(result.lower.marginals, dtype=float)
    upper_marginals = np.asarray(result.upper.marginals, dtype=float)
    lower_dual = np.maximum(lower_marginals, 0.0)
    upper_dual = np.maximum(-upper_marginals, 0.0)

    return FollowerSolution(
        quantity=q,
        retailer_profit=retailer_profit,
        capacity_dual=capacity_dual,
        upper_dual=upper_dual,
        lower_dual=lower_dual,
    )


def follower_objective(
    instance: SupplyChainInstance,
    wholesale_price: np.ndarray,
    quantity: np.ndarray,
) -> float:
    w = np.asarray(wholesale_price, dtype=float)
    q = np.asarray(quantity, dtype=float)
    return float(np.dot(instance.retail_price - w, q))


def follower_feasible(instance: SupplyChainInstance, quantity: np.ndarray, tol: float = 1e-8) -> bool:
    q = np.asarray(quantity, dtype=float)
    if q.shape != instance.retail_price.shape:
        return False
    if np.any(q < -tol) or np.any(q > instance.quantity_cap + tol):
        return False
    return bool(np.dot(instance.resource_use, q) <= instance.retailer_capacity + tol)
