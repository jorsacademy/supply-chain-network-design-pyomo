from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

import pulp

from robust_procurement_model import build_sample_data as build_box_sample_data


@dataclass(frozen=True)
class BudgetedRobustArtifacts:
    model: pulp.LpProblem
    procurement: dict
    raw_inventory: dict
    production: dict
    setup: dict
    plant_inventory: dict
    dc_inventory: dict
    plant_to_dc: dict
    dc_to_market: dict
    cost_terms: dict[str, pulp.LpAffineExpression]


def build_sample_data() -> dict[str, Any]:
    """Return a deterministic instance with Bertsimas-Sim demand budgets."""
    data = build_box_sample_data()
    data["uncertainty_budget"] = {
        t: {product: 1.0 for product in data["products"]}
        for t in data["periods"]
    }
    return data


def budgeted_protection(deviations: list[float], gamma: float) -> float:
    """Return the Bertsimas-Sim protection value for one uncertainty budget.

    For non-negative deviations d_j and budget Gamma, the protection term is
    the sum of the largest floor(Gamma) deviations plus the corresponding
    fraction of the next-largest deviation.
    """
    if gamma < 0 or gamma > len(deviations):
        raise ValueError("Gamma must lie between zero and the number of uncertain components.")
    if any(value < 0 for value in deviations):
        raise ValueError("Demand deviations must be non-negative.")

    ordered = sorted(deviations, reverse=True)
    integer_part = floor(gamma)
    fractional_part = gamma - integer_part
    protection = sum(ordered[:integer_part])
    if fractional_part > 0 and integer_part < len(ordered):
        protection += fractional_part * ordered[integer_part]
    return protection


def protected_total_demand(data: dict[str, Any], t: int, product: str) -> float:
    """Return aggregate nominal demand plus Bertsimas-Sim protection."""
    deviations = [
        data["demand_deviation"][t][market][product]
        for market in data["markets"]
    ]
    nominal = sum(
        data["nominal_demand"][t][market][product]
        for market in data["markets"]
    )
    gamma = data["uncertainty_budget"][t][product]
    return nominal + budgeted_protection(deviations, gamma)


def validate_data(data: dict[str, Any]) -> None:
    """Validate the dimensions and assumptions of the budgeted robust model."""
    required = {
        "periods",
        "suppliers",
        "plants",
        "dcs",
        "markets",
        "products",
        "raw_materials",
        "nominal_demand",
        "demand_deviation",
        "uncertainty_budget",
        "bill_of_materials",
        "supplier_capacity",
        "procurement_cost",
        "supplier_to_plant_cost",
        "procurement_lead_time",
        "initial_raw_inventory",
        "raw_inventory_capacity",
        "raw_holding_cost",
        "production_capacity",
        "type_capacity",
        "manufacturing_cost",
        "setup_cost",
        "plant_holding_cost",
        "dc_holding_cost",
        "plant_to_dc_cost",
        "dc_to_market_cost",
        "distribution_lead_time",
        "initial_plant_inventory",
        "initial_dc_inventory",
        "plant_inventory_capacity",
        "dc_inventory_capacity",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing data sections: {sorted(missing)}")

    periods = data["periods"]
    if not periods or periods != list(range(periods[0], periods[-1] + 1)):
        raise ValueError("Periods must be a non-empty consecutive integer sequence.")

    for t in periods:
        for product in data["products"]:
            gamma = data["uncertainty_budget"][t][product]
            if gamma < 0 or gamma > len(data["markets"]):
                raise ValueError(
                    "Each uncertainty budget must lie between zero and the number of markets."
                )
            for market in data["markets"]:
                nominal = data["nominal_demand"][t][market][product]
                deviation = data["demand_deviation"][t][market][product]
                if nominal < 0 or deviation < 0:
                    raise ValueError("Demand and demand deviations must be non-negative.")

    for product in data["products"]:
        for material in data["raw_materials"]:
            if data["bill_of_materials"][product][material] < 0:
                raise ValueError("Bill-of-material coefficients must be non-negative.")

    for supplier in data["suppliers"]:
        for plant in data["plants"]:
            lead = data["procurement_lead_time"][supplier][plant]
            if not isinstance(lead, int) or lead < 1:
                raise ValueError("Procurement lead times must be positive integers.")

    for plant in data["plants"]:
        for dc in data["dcs"]:
            lead = data["distribution_lead_time"][plant][dc]
            if not isinstance(lead, int) or lead < 1:
                raise ValueError("Distribution lead times must be positive integers.")

    first = periods[0]
    for product in data["products"]:
        required_first = protected_total_demand(data, first, product)
        available_first = sum(
            data["initial_dc_inventory"][dc][product] for dc in data["dcs"]
        )
        if available_first < required_first:
            raise ValueError(
                f"Initial DC inventory for {product} is insufficient for first-period budgeted robust demand."
            )


def build_model(data: dict[str, Any]) -> BudgetedRobustArtifacts:
    """Build a procurement MILP with Bertsimas-Sim budgeted demand uncertainty.

    The uncertainty budget is applied across market demand deviations for each
    period-product pair. Every market receives at least nominal demand, while
    aggregate deliveries include the Bertsimas-Sim protection term.
    """
    validate_data(data)

    periods = data["periods"]
    suppliers = data["suppliers"]
    plants = data["plants"]
    dcs = data["dcs"]
    markets = data["markets"]
    products = data["products"]
    materials = data["raw_materials"]

    model = pulp.LpProblem(
        "Budgeted_Robust_Bioreactor_Supply_Chain_Optimization", pulp.LpMinimize
    )

    procurement = pulp.LpVariable.dicts(
        "Procurement",
        [(t, s, p, r) for t in periods for s in suppliers for p in plants for r in materials],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    raw_inventory = pulp.LpVariable.dicts(
        "RawInventory",
        [(t, p, r) for t in periods for p in plants for r in materials],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    production = pulp.LpVariable.dicts(
        "Production",
        [(t, p, k) for t in periods for p in plants for k in products],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    setup = pulp.LpVariable.dicts(
        "Setup",
        [(t, p, k) for t in periods for p in plants for k in products],
        cat=pulp.LpBinary,
    )
    plant_inventory = pulp.LpVariable.dicts(
        "PlantInventory",
        [(t, p, k) for t in periods for p in plants for k in products],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    dc_inventory = pulp.LpVariable.dicts(
        "DCInventory",
        [(t, d, k) for t in periods for d in dcs for k in products],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    plant_to_dc = pulp.LpVariable.dicts(
        "PlantToDC",
        [(t, p, d, k) for t in periods for p in plants for d in dcs for k in products],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    dc_to_market = pulp.LpVariable.dicts(
        "DCToMarket",
        [(t, d, m, k) for t in periods for d in dcs for m in markets for k in products],
        lowBound=0,
        cat=pulp.LpContinuous,
    )

    cost_terms = {
        "procurement": pulp.lpSum(
            data["procurement_cost"][s][r] * procurement[(t, s, p, r)]
            for t in periods for s in suppliers for p in plants for r in materials
        ),
        "raw_material_transport": pulp.lpSum(
            data["supplier_to_plant_cost"][s][p][r] * procurement[(t, s, p, r)]
            for t in periods for s in suppliers for p in plants for r in materials
        ),
        "raw_material_inventory": pulp.lpSum(
            data["raw_holding_cost"][p][r] * raw_inventory[(t, p, r)]
            for t in periods for p in plants for r in materials
        ),
        "manufacturing": pulp.lpSum(
            data["manufacturing_cost"][p][k] * production[(t, p, k)]
            for t in periods for p in plants for k in products
        ),
        "setup": pulp.lpSum(
            data["setup_cost"][p][k] * setup[(t, p, k)]
            for t in periods for p in plants for k in products
        ),
        "finished_goods_inventory": pulp.lpSum(
            data["plant_holding_cost"][p] * plant_inventory[(t, p, k)]
            for t in periods for p in plants for k in products
        ) + pulp.lpSum(
            data["dc_holding_cost"][d] * dc_inventory[(t, d, k)]
            for t in periods for d in dcs for k in products
        ),
        "distribution_transport": pulp.lpSum(
            data["plant_to_dc_cost"][p][d] * plant_to_dc[(t, p, d, k)]
            for t in periods for p in plants for d in dcs for k in products
        ) + pulp.lpSum(
            data["dc_to_market_cost"][d][m] * dc_to_market[(t, d, m, k)]
            for t in periods for d in dcs for m in markets for k in products
        ),
    }
    model += pulp.lpSum(cost_terms.values()), "TotalCost"

    for t in periods:
        for supplier in suppliers:
            for material in materials:
                model += (
                    pulp.lpSum(procurement[(t, supplier, plant, material)] for plant in plants)
                    <= data["supplier_capacity"][t][supplier][material],
                    f"SupplierCapacity_{t}_{supplier}_{material}",
                )

    for t in periods:
        for plant in plants:
            for material in materials:
                previous = (
                    data["initial_raw_inventory"][plant][material]
                    if t == periods[0]
                    else raw_inventory[(t - 1, plant, material)]
                )
                arrivals = pulp.lpSum(
                    procurement[(depart_t, supplier, plant, material)]
                    for supplier in suppliers
                    for depart_t in periods
                    if depart_t + data["procurement_lead_time"][supplier][plant] == t
                )
                consumption = pulp.lpSum(
                    data["bill_of_materials"][product][material]
                    * production[(t, plant, product)]
                    for product in products
                )
                model += (
                    raw_inventory[(t, plant, material)]
                    == previous + arrivals - consumption,
                    f"RawBalance_{t}_{plant}_{material}",
                )

            model += (
                pulp.lpSum(raw_inventory[(t, plant, material)] for material in materials)
                <= data["raw_inventory_capacity"][plant],
                f"RawInventoryCapacity_{t}_{plant}",
            )

            model += (
                pulp.lpSum(production[(t, plant, product)] for product in products)
                <= data["production_capacity"][plant],
                f"ProductionCapacity_{t}_{plant}",
            )
            for product in products:
                model += (
                    production[(t, plant, product)]
                    <= data["type_capacity"][plant][product] * setup[(t, plant, product)],
                    f"SetupLink_{t}_{plant}_{product}",
                )

    for t in periods:
        for plant in plants:
            for product in products:
                previous = (
                    data["initial_plant_inventory"][plant][product]
                    if t == periods[0]
                    else plant_inventory[(t - 1, plant, product)]
                )
                outbound = pulp.lpSum(plant_to_dc[(t, plant, dc, product)] for dc in dcs)
                model += (
                    plant_inventory[(t, plant, product)]
                    == previous + production[(t, plant, product)] - outbound,
                    f"PlantBalance_{t}_{plant}_{product}",
                )

            model += (
                pulp.lpSum(plant_inventory[(t, plant, product)] for product in products)
                <= data["plant_inventory_capacity"][plant],
                f"PlantInventoryCapacity_{t}_{plant}",
            )

    for t in periods:
        for dc in dcs:
            for product in products:
                previous = (
                    data["initial_dc_inventory"][dc][product]
                    if t == periods[0]
                    else dc_inventory[(t - 1, dc, product)]
                )
                arrivals = pulp.lpSum(
                    plant_to_dc[(depart_t, plant, dc, product)]
                    for plant in plants
                    for depart_t in periods
                    if depart_t + data["distribution_lead_time"][plant][dc] == t
                )
                outbound = pulp.lpSum(
                    dc_to_market[(t, dc, market, product)] for market in markets
                )
                model += (
                    dc_inventory[(t, dc, product)] == previous + arrivals - outbound,
                    f"DCBalance_{t}_{dc}_{product}",
                )

            model += (
                pulp.lpSum(dc_inventory[(t, dc, product)] for product in products)
                <= data["dc_inventory_capacity"][dc],
                f"DCInventoryCapacity_{t}_{dc}",
            )

    for t in periods:
        for product in products:
            for market in markets:
                delivered = pulp.lpSum(
                    dc_to_market[(t, dc, market, product)] for dc in dcs
                )
                model += (
                    delivered >= data["nominal_demand"][t][market][product],
                    f"NominalDemand_{t}_{market}_{product}",
                )

            model += (
                pulp.lpSum(
                    dc_to_market[(t, dc, market, product)]
                    for dc in dcs
                    for market in markets
                )
                == protected_total_demand(data, t, product),
                f"BudgetedRobustDemand_{t}_{product}",
            )

    last_period = periods[-1]
    for t in periods:
        for supplier in suppliers:
            for plant in plants:
                if t + data["procurement_lead_time"][supplier][plant] > last_period:
                    for material in materials:
                        model += (
                            procurement[(t, supplier, plant, material)] == 0,
                            f"NoLateProcurement_{t}_{supplier}_{plant}_{material}",
                        )

        for plant in plants:
            for dc in dcs:
                if t + data["distribution_lead_time"][plant][dc] > last_period:
                    for product in products:
                        model += (
                            plant_to_dc[(t, plant, dc, product)] == 0,
                            f"NoLateDistribution_{t}_{plant}_{dc}_{product}",
                        )

    return BudgetedRobustArtifacts(
        model=model,
        procurement=procurement,
        raw_inventory=raw_inventory,
        production=production,
        setup=setup,
        plant_inventory=plant_inventory,
        dc_inventory=dc_inventory,
        plant_to_dc=plant_to_dc,
        dc_to_market=dc_to_market,
        cost_terms=cost_terms,
    )


def solve_model(data: dict[str, Any], solver_msg: bool = False) -> dict[str, Any]:
    """Solve the budgeted robust MILP and return validated decision values."""
    artifacts = build_model(data)
    status_code = artifacts.model.solve(pulp.PULP_CBC_CMD(msg=solver_msg))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        raise RuntimeError(
            f"Optimization did not produce an optimal solution. Solver status: {status}"
        )

    def values(variable_dict: dict) -> dict:
        return {
            key: float(variable.value())
            for key, variable in variable_dict.items()
        }

    cost_breakdown = {
        name: float(pulp.value(expression))
        for name, expression in artifacts.cost_terms.items()
    }
    total_cost = float(pulp.value(artifacts.model.objective))
    if abs(sum(cost_breakdown.values()) - total_cost) > 1e-6:
        raise RuntimeError("Objective decomposition does not match the model objective.")

    return {
        "status": status,
        "total_cost": total_cost,
        "cost_breakdown": cost_breakdown,
        "procurement": values(artifacts.procurement),
        "raw_inventory": values(artifacts.raw_inventory),
        "production": values(artifacts.production),
        "setup": values(artifacts.setup),
        "plant_inventory": values(artifacts.plant_inventory),
        "dc_inventory": values(artifacts.dc_inventory),
        "plant_to_dc": values(artifacts.plant_to_dc),
        "dc_to_market": values(artifacts.dc_to_market),
    }


def print_summary(data: dict[str, Any], results: dict[str, Any]) -> None:
    """Print the uncertainty budget and solved objective."""
    print("Bertsimas-Sim Budgeted Robust Procurement Model")
    print("-----------------------------------------------")
    print(f"Status: {results['status']}")
    print(f"Total cost: {results['total_cost']:,.2f}")
    print("Uncertainty budgets:")
    for t in data["periods"]:
        values = ", ".join(
            f"{product}={data['uncertainty_budget'][t][product]:g}"
            for product in data["products"]
        )
        print(f"  Period {t}: {values}")


def main() -> None:
    data = build_sample_data()
    results = solve_model(data)
    print_summary(data, results)


if __name__ == "__main__":
    main()
