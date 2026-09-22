from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupplyChainInstance:
    """Small linear Stackelberg pricing instance.

    The manufacturer (leader) chooses wholesale prices ``w``. The retailer
    (follower) chooses purchase quantities ``q`` to maximize retail margin
    subject to one shared handling-capacity constraint and item-specific caps.
    """

    retail_price: np.ndarray
    production_cost: np.ndarray
    resource_use: np.ndarray
    retailer_capacity: float
    quantity_cap: np.ndarray
    wholesale_lower: np.ndarray
    wholesale_upper: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.retail_price,
            self.production_cost,
            self.resource_use,
            self.quantity_cap,
            self.wholesale_lower,
            self.wholesale_upper,
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("all product data must be one-dimensional")
        n = self.retail_price.size
        if n < 1 or any(array.size != n for array in arrays):
            raise ValueError("all product vectors must have the same positive length")
        if not np.all(np.isfinite(np.concatenate(arrays))):
            raise ValueError("all product data must be finite")
        if not np.isfinite(self.retailer_capacity) or self.retailer_capacity <= 0:
            raise ValueError("retailer_capacity must be positive and finite")
        if np.any(self.resource_use <= 0):
            raise ValueError("resource_use must be strictly positive")
        if np.any(self.quantity_cap <= 0):
            raise ValueError("quantity_cap must be strictly positive")
        if np.any(self.wholesale_lower > self.wholesale_upper):
            raise ValueError("wholesale price bounds are inconsistent")

    @property
    def n_products(self) -> int:
        return int(self.retail_price.size)


@dataclass(frozen=True)
class FollowerSolution:
    quantity: np.ndarray
    retailer_profit: float
    capacity_dual: float
    upper_dual: np.ndarray
    lower_dual: np.ndarray


@dataclass(frozen=True)
class BilevelSolution:
    wholesale_price: np.ndarray
    quantity: np.ndarray
    manufacturer_profit: float
    retailer_profit: float
    capacity_dual: float
    upper_dual: np.ndarray
    lower_dual: np.ndarray
    status: str

    @property
    def channel_profit(self) -> float:
        return self.manufacturer_profit + self.retailer_profit


def demo_instance() -> SupplyChainInstance:
    """Return the reproducible three-product teaching instance."""

    return SupplyChainInstance(
        retail_price=np.asarray([18.0, 15.0, 13.0]),
        production_cost=np.asarray([5.0, 4.0, 3.0]),
        resource_use=np.asarray([3.0, 2.0, 1.5]),
        retailer_capacity=27.0,
        quantity_cap=np.asarray([5.0, 8.0, 10.0]),
        wholesale_lower=np.asarray([6.0, 5.0, 4.0]),
        wholesale_upper=np.asarray([16.0, 13.0, 11.0]),
    )


def random_instance(n_products: int = 3, seed: int = 0) -> SupplyChainInstance:
    """Generate a small instance suitable for verification experiments."""

    if n_products < 1:
        raise ValueError("n_products must be positive")
    rng = np.random.default_rng(seed)
    production_cost = rng.uniform(2.0, 7.0, size=n_products)
    retail_price = production_cost + rng.uniform(6.0, 14.0, size=n_products)
    resource_use = rng.uniform(1.0, 4.0, size=n_products)
    quantity_cap = rng.uniform(3.0, 10.0, size=n_products)
    retailer_capacity = float(0.55 * np.dot(resource_use, quantity_cap))
    wholesale_lower = production_cost + rng.uniform(0.2, 1.5, size=n_products)
    wholesale_upper = retail_price + rng.uniform(0.5, 4.0, size=n_products)
    return SupplyChainInstance(
        retail_price=retail_price,
        production_cost=production_cost,
        resource_use=resource_use,
        retailer_capacity=retailer_capacity,
        quantity_cap=quantity_cap,
        wholesale_lower=wholesale_lower,
        wholesale_upper=wholesale_upper,
    )
