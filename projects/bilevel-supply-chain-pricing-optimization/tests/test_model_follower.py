import numpy as np
import pytest

from bilevel_pricing import (
    SupplyChainInstance,
    demo_instance,
    follower_feasible,
    solve_follower,
)


def test_instance_validation_rejects_inconsistent_bounds() -> None:
    with pytest.raises(ValueError):
        SupplyChainInstance(
            retail_price=np.asarray([10.0]),
            production_cost=np.asarray([2.0]),
            resource_use=np.asarray([1.0]),
            retailer_capacity=5.0,
            quantity_cap=np.asarray([4.0]),
            wholesale_lower=np.asarray([9.0]),
            wholesale_upper=np.asarray([8.0]),
        )


def test_follower_solution_is_feasible() -> None:
    instance = demo_instance()
    follower = solve_follower(instance, np.asarray([12.0, 10.0, 9.0]))
    assert follower_feasible(instance, follower.quantity)
    assert follower.retailer_profit >= -1e-9


def test_one_product_follower_is_hand_checkable() -> None:
    instance = SupplyChainInstance(
        retail_price=np.asarray([12.0]),
        production_cost=np.asarray([3.0]),
        resource_use=np.asarray([2.0]),
        retailer_capacity=8.0,
        quantity_cap=np.asarray([10.0]),
        wholesale_lower=np.asarray([4.0]),
        wholesale_upper=np.asarray([14.0]),
    )
    follower = solve_follower(instance, np.asarray([8.0]))
    assert follower.quantity[0] == pytest.approx(4.0)
    assert follower.retailer_profit == pytest.approx(16.0)
