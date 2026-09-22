from copy import deepcopy

import pytest

from bioreactor_supply_chain import (
    build_sample_data,
    results_to_frames,
    solve_model,
    validate_data,
)


def test_sample_instance_solves_to_optimality():
    data = build_sample_data()
    results = solve_model(data)

    assert results["status"] == "Optimal"
    assert results["total_cost"] > 0
    assert sum(results["production"].values()) > 0


def test_demand_is_satisfied_exactly():
    data = build_sample_data()
    results = solve_model(data)

    for period in data["periods"]:
        for market in data["markets"]:
            for product in data["products"]:
                delivered = sum(
                    results["dc_to_market"][(period, dc, market, product)]
                    for dc in data["dcs"]
                )
                assert delivered == data["demand"][period][market][product]


def test_dc_inventory_balance_respects_lead_times():
    data = build_sample_data()
    results = solve_model(data)

    for period in data["periods"]:
        for dc in data["dcs"]:
            for product in data["products"]:
                previous = (
                    data["initial_dc_inventory"][dc][product]
                    if period == data["periods"][0]
                    else results["dc_inventory"][(period - 1, dc, product)]
                )
                arriving = sum(
                    results["plant_to_dc"][(depart_period, plant, dc, product)]
                    for plant in data["plants"]
                    for depart_period in data["periods"]
                    if depart_period + data["lead_time"][plant][dc] == period
                )
                outbound = sum(
                    results["dc_to_market"][(period, dc, market, product)]
                    for market in data["markets"]
                )
                expected = previous + arriving - outbound
                assert results["dc_inventory"][(period, dc, product)] == expected


def test_cost_breakdown_matches_total_objective():
    results = solve_model(build_sample_data())
    assert sum(results["cost_breakdown"].values()) == pytest.approx(
        results["total_cost"], abs=1e-6
    )


def test_result_frames_keep_zero_decisions():
    data = build_sample_data()
    results = solve_model(data)
    frames = results_to_frames(results)

    expected_production_rows = (
        len(data["periods"]) * len(data["plants"]) * len(data["products"])
    )
    assert len(frames["production"]) == expected_production_rows
    assert (frames["production"]["Production"] == 0).any()


def test_invalid_first_period_inventory_is_rejected():
    data = deepcopy(build_sample_data())
    for dc in data["dcs"]:
        for product in data["products"]:
            data["initial_dc_inventory"][dc][product] = 0

    with pytest.raises(ValueError, match="insufficient for first-period demand"):
        validate_data(data)
