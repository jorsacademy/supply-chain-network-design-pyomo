from copy import deepcopy

import pytest

from budgeted_robust_procurement_model import (
    budgeted_protection,
    build_sample_data,
    protected_total_demand,
    solve_model,
    validate_data,
)


def test_budgeted_protection_integer_gamma():
    assert budgeted_protection([4.0, 2.0, 1.0], 2.0) == pytest.approx(6.0)


def test_budgeted_protection_fractional_gamma():
    assert budgeted_protection([4.0, 2.0, 1.0], 1.5) == pytest.approx(5.0)


def test_budgeted_protection_gamma_zero():
    assert budgeted_protection([4.0, 2.0, 1.0], 0.0) == pytest.approx(0.0)


def test_budgeted_model_solves_to_optimality():
    data = build_sample_data()
    results = solve_model(data)
    assert results["status"] == "Optimal"
    assert results["total_cost"] > 0


def test_nominal_market_demand_is_respected():
    data = build_sample_data()
    results = solve_model(data)

    for period in data["periods"]:
        for market in data["markets"]:
            for product in data["products"]:
                delivered = sum(
                    results["dc_to_market"][(period, dc, market, product)]
                    for dc in data["dcs"]
                )
                assert delivered + 1e-6 >= data["nominal_demand"][period][market][product]


def test_aggregate_delivery_matches_budgeted_protection():
    data = build_sample_data()
    results = solve_model(data)

    for period in data["periods"]:
        for product in data["products"]:
            delivered = sum(
                results["dc_to_market"][(period, dc, market, product)]
                for dc in data["dcs"]
                for market in data["markets"]
            )
            assert delivered == pytest.approx(
                protected_total_demand(data, period, product), abs=1e-6
            )


def test_gamma_monotonicity_of_protection():
    data = build_sample_data()
    period = data["periods"][1]
    product = data["products"][0]

    low = deepcopy(data)
    high = deepcopy(data)
    low["uncertainty_budget"][period][product] = 0.0
    high["uncertainty_budget"][period][product] = float(len(data["markets"]))

    assert protected_total_demand(high, period, product) >= protected_total_demand(
        low, period, product
    )


def test_invalid_gamma_is_rejected():
    data = deepcopy(build_sample_data())
    period = data["periods"][0]
    product = data["products"][0]
    data["uncertainty_budget"][period][product] = len(data["markets"]) + 0.1

    with pytest.raises(ValueError, match="uncertainty budget"):
        validate_data(data)


def test_cost_breakdown_matches_total_objective():
    results = solve_model(build_sample_data())
    assert sum(results["cost_breakdown"].values()) == pytest.approx(
        results["total_cost"], abs=1e-6
    )
