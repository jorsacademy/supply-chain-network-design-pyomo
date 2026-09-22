from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linprog

from .exact import solve_exact_optimistic
from .kkt import solve_kkt_milp
from .model import SupplyChainInstance, demo_instance


@dataclass(frozen=True)
class BenchmarkResult:
    kkt_manufacturer_profit: float
    exact_manufacturer_profit: float
    absolute_profit_gap: float
    retailer_profit: float
    channel_profit: float
    centralized_channel_profit: float
    channel_efficiency: float
    wholesale_price: tuple[float, ...]
    quantity: tuple[float, ...]
    capacity_dual: float
    exact_extreme_points: int
    exact_leader_lps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def centralized_channel_optimum(instance: SupplyChainInstance) -> float:
    """Return the integrated channel-profit upper benchmark."""

    result = linprog(
        c=-(instance.retail_price - instance.production_cost),
        A_ub=instance.resource_use.reshape(1, -1),
        b_ub=np.asarray([instance.retailer_capacity]),
        bounds=[(0.0, float(cap)) for cap in instance.quantity_cap],
        method="highs",
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"centralized LP failed: {result.message}")
    return float(-result.fun)


def run_benchmark(instance: SupplyChainInstance | None = None) -> BenchmarkResult:
    """Cross-check the KKT MILP against the independent exact bilevel oracle."""

    instance = demo_instance() if instance is None else instance
    kkt = solve_kkt_milp(instance)
    exact = solve_exact_optimistic(instance)
    centralized = centralized_channel_optimum(instance)
    channel_efficiency = 1.0 if centralized <= 1e-12 else kkt.channel_profit / centralized
    return BenchmarkResult(
        kkt_manufacturer_profit=kkt.manufacturer_profit,
        exact_manufacturer_profit=exact.manufacturer_profit,
        absolute_profit_gap=abs(kkt.manufacturer_profit - exact.manufacturer_profit),
        retailer_profit=kkt.retailer_profit,
        channel_profit=kkt.channel_profit,
        centralized_channel_profit=centralized,
        channel_efficiency=channel_efficiency,
        wholesale_price=tuple(float(value) for value in kkt.wholesale_price),
        quantity=tuple(float(value) for value in kkt.quantity),
        capacity_dual=kkt.capacity_dual,
        exact_extreme_points=exact.extreme_points_checked,
        exact_leader_lps=exact.leader_lps_solved,
    )
