import numpy as np
import pytest

from bilevel_pricing import (
    SupplyChainInstance,
    demo_instance,
    derive_complementarity_bounds,
    solve_follower,
    solve_kkt_milp,
)


def test_derived_big_m_bounds_cover_demo_duals() -> None:
    instance = demo_instance()
    solution = solve_kkt_milp(instance)
    bounds = derive_complementarity_bounds(instance)
    assert 0 <= solution.capacity_dual <= bounds.capacity_dual + 1e-8
    assert np.all(solution.upper_dual <= bounds.upper_dual + 1e-8)
    assert np.all(solution.lower_dual <= bounds.lower_dual + 1e-8)


def test_kkt_quantity_is_follower_optimal() -> None:
    instance = demo_instance()
    solution = solve_kkt_milp(instance)
    follower = solve_follower(instance, solution.wholesale_price)
    direct_value = np.dot(instance.retail_price - solution.wholesale_price, solution.quantity)
    assert direct_value == pytest.approx(follower.retailer_profit, abs=1e-6)


def test_one_product_bilevel_solution_matches_analytic_optimistic_case() -> None:
    instance = SupplyChainInstance(
        retail_price=np.asarray([12.0]),
        production_cost=np.asarray([3.0]),
        resource_use=np.asarray([2.0]),
        retailer_capacity=8.0,
        quantity_cap=np.asarray([10.0]),
        wholesale_lower=np.asarray([4.0]),
        wholesale_upper=np.asarray([14.0]),
    )
    solution = solve_kkt_milp(instance)
    assert solution.wholesale_price[0] == pytest.approx(12.0, abs=1e-6)
    assert solution.quantity[0] == pytest.approx(4.0, abs=1e-6)
    assert solution.manufacturer_profit == pytest.approx(36.0, abs=1e-6)
