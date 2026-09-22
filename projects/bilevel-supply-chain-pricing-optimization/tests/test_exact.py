import numpy as np
import pytest

from bilevel_pricing import (
    demo_instance,
    enumerate_follower_extreme_points,
    follower_feasible,
    random_instance,
    solve_exact_optimistic,
    solve_kkt_milp,
)


def test_extreme_points_are_feasible_and_unique() -> None:
    instance = demo_instance()
    points = enumerate_follower_extreme_points(instance)
    assert len(points) >= 4
    assert all(follower_feasible(instance, point) for point in points)
    for i, left in enumerate(points):
        for right in points[i + 1 :]:
            assert not np.allclose(left, right, atol=1e-8)


def test_demo_kkt_matches_independent_exact_oracle() -> None:
    instance = demo_instance()
    kkt = solve_kkt_milp(instance)
    exact = solve_exact_optimistic(instance)
    assert kkt.manufacturer_profit == pytest.approx(exact.manufacturer_profit, abs=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_random_small_instances_match_exact_oracle(seed: int) -> None:
    instance = random_instance(n_products=3, seed=seed)
    kkt = solve_kkt_milp(instance)
    exact = solve_exact_optimistic(instance)
    assert kkt.manufacturer_profit == pytest.approx(exact.manufacturer_profit, abs=1e-6)
