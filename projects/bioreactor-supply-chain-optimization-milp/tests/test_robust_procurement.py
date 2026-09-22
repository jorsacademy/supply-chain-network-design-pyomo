from copy import deepcopy

import pytest

from robust_procurement_model import build_sample_data, robust_demand, solve_model, validate_data


def test_robust_procurement_instance_solves_to_optimality():
    data = build_sample_data()
    results = solve_model(data)

    assert results["status"] == "Optimal"
    assert results["total_cost"] > 0
    assert sum(results["procurement"].values()) > 0
    assert sum(results["production"].values()) > 0


def test_market_deliveries_cover_box_uncertainty_upper_bound():
    data = build_sample_data()
    results = solve_model(data)

    for period in data["periods"]:
        for market in data["markets"]:
            for product in data["products"]:
                delivered = sum(
                    results["dc_to_market"][(period, dc, market, product)]
                    for dc in data["dcs"]
                )
                assert delivered >= robust_demand(data, period, market, product)


def test_raw_material_balance_respects_procurement_lead_times_and_bom():
    data = build_sample_data()
    results = solve_model(data)

    for period in data["periods"]:
        for plant in data["plants"]:
            for material in data["raw_materials"]:
                previous = (
                    data["initial_raw_inventory"][plant][material]
                    if period == data["periods"][0]
                    else results["raw_inventory"][(period - 1, plant, material)]
                )
                arrivals = sum(
                    results["procurement"][(depart_period, supplier, plant, material)]
                    for supplier in data["suppliers"]
                    for depart_period in data["periods"]
                    if depart_period + data["procurement_lead_time"][supplier][plant]
                    == period
                )
                consumption = sum(
                    data["bill_of_materials"][product][material]
                    * results["production"][(period, plant, product)]
                    for product in data["products"]
                )
                expected = previous + arrivals - consumption
                assert results["raw_inventory"][(period, plant, material)] == expected


def test_cost_breakdown_matches_total_objective():
    results = solve_model(build_sample_data())
    assert sum(results["cost_breakdown"].values()) == pytest.approx(
        results["total_cost"], abs=1e-6
    )


def test_negative_demand_deviation_is_rejected():
    data = deepcopy(build_sample_data())
    data["demand_deviation"][1]["Market_1"]["Small_Bioreactor"] = -1

    with pytest.raises(ValueError, match="Demand and demand deviations must be non-negative"):
        validate_data(data)


def test_insufficient_first_period_robust_inventory_is_rejected():
    data = deepcopy(build_sample_data())
    for dc in data["dcs"]:
        for product in data["products"]:
            data["initial_dc_inventory"][dc][product] = 0

    with pytest.raises(ValueError, match="insufficient for robust first-period demand"):
        validate_data(data)
