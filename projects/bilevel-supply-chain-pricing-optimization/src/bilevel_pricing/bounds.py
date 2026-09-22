from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import SupplyChainInstance


@dataclass(frozen=True)
class ComplementarityBounds:
    capacity_slack: float
    capacity_dual: float
    upper_slack: np.ndarray
    upper_dual: np.ndarray
    lower_slack: np.ndarray
    lower_dual: np.ndarray


def derive_complementarity_bounds(instance: SupplyChainInstance) -> ComplementarityBounds:
    """Derive structure-specific valid bounds for the KKT disjunctions.

    The follower is a continuous knapsack LP with one shared resource row. Let
    m_j = retail_price_j - wholesale_price_j. A dual-optimal capacity multiplier
    can always be chosen no larger than max_j max(0, m_j) / resource_use_j.
    The remaining multiplier bounds then follow from stationarity and bound
    complementarity. These are problem-derived constants, not trial-and-error M values.
    """

    margin_max = instance.retail_price - instance.wholesale_lower
    margin_min = instance.retail_price - instance.wholesale_upper
    lambda_max = float(
        max(0.0, np.max(np.maximum(margin_max, 0.0) / instance.resource_use))
    )
    mu_max = np.maximum(margin_max, 0.0)
    nu_max = np.maximum(instance.resource_use * lambda_max - margin_min, 0.0)
    return ComplementarityBounds(
        capacity_slack=float(instance.retailer_capacity),
        capacity_dual=lambda_max,
        upper_slack=instance.quantity_cap.copy(),
        upper_dual=mu_max,
        lower_slack=instance.quantity_cap.copy(),
        lower_dual=nu_max,
    )
