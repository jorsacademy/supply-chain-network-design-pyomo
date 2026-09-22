from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pulp


@dataclass(frozen=True)
class RobustArtifacts:
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
    """Return a deterministic synthetic instance with box-uncertain demand."""
    periods = [1, 2, 3, 4, 5]
    suppliers = ["Supplier_1", "Supplier_2"]
    plants = ["Plant_1", "Plant_2"]
    dcs = ["DC_1", "DC_2"]
    markets = ["Market_1", "Market_2"]
    products = ["Small_Bioreactor", "Large_Bioreactor"]
    raw_materials = ["Structural_Module", "Control_Module"]

    nominal_demand = {
        1: {
            "Market_1": {"Small_Bioreactor": 3, "Large_Bioreactor": 1},
            "Market_2": {"Small_Bioreactor": 2, "Large_Bioreactor": 1},
        },
        2: {
            "Market_1": {"Small_Bioreactor": 4, "Large_Bioreactor": 2},
            "Market_2": {"Small_Bioreactor": 3, "Large_Bioreactor": 1},
        },
        3: {
            "Market_1": {"Small_Bioreactor": 5, "Large_Bioreactor": 2},
            "Market_2": {"Small_Bioreactor": 3, "Large_Bioreactor": 2},
        },
        4: {
            "Market_1": {"Small_Bioreactor": 4, "Large_Bioreactor": 2},
            "Market_2": {"Small_Bioreactor": 4, "Large_Bioreactor": 1},
        },
        5: {
            "Market_1": {"Small_Bioreactor": 4, "Large_Bioreactor": 2},
            "Market_2": {"Small_Bioreactor": 3, "Large_Bioreactor": 1},
        },
    }

    demand_deviation = {
        t: {
            market: {
                "Small_Bioreactor": 1,
                "Large_Bioreactor": 1 if market == "Market_1" else 0,
            }
            for market in markets
        }
        for t in periods
    }

    return {
        "periods": periods,
        "suppliers": suppliers,
        "plants": plants,
        "dcs": dcs,
        "markets": markets,
        "products": products,
        "raw_materials": raw_materials,
        "nominal_demand": nominal_demand,
        "demand_deviation": demand_deviation,
        "bill_of_materials": {
            "Small_Bioreactor": {"Structural_Module": 2, "Control_Module": 1},
            "Large_Bioreactor": {"Structural_Module": 4, "Control_Module": 2},
        },
        "supplier_capacity": {
            t: {
                "Supplier_1": {"Structural_Module": 45, "Control_Module": 25},
                "Supplier_2": {"Structural_Module": 40, "Control_Module": 25},
            }
            for t in periods
        },
        "procurement_cost": {
            "Supplier_1": {"Structural_Module": 9, "Control_Module": 14},
            "Supplier_2": {"Structural_Module": 10, "Control_Module": 13},
        },
        "supplier_to_plant_cost": {
            "Supplier_1": {
                "Plant_1": {"Structural_Module": 2, "Control_Module": 2},
                "Plant_2": {"Structural_Module": 4, "Control_Module": 3},
            },
            "Supplier_2": {
                "Plant_1": {"Structural_Module": 4, "Control_Module": 3},
                "Plant_2": {"Structural_Module": 2, "Control_Module": 2},
            },
        },
        "procurement_lead_time": {
            "Supplier_1": {"Plant_1": 1, "Plant_2": 1},
            "Supplier_2": {"Plant_1": 1, "Plant_2": 1},
        },
        "initial_raw_inventory": {
            "Plant_1": {"Structural_Module": 35, "Control_Module": 20},
            "Plant_2": {"Structural_Module": 35, "Control_Module": 20},
        },
        "raw_inventory_capacity": {"Plant_1": 100, "Plant_2": 100},
        "raw_holding_cost": {
            "Plant_1": {"Structural_Module": 1.0, "Control_Module": 1.5},
            "Plant_2": {"Structural_Module": 1.0, "Control_Module": 1.5},
        },
        "production_capacity": {"Plant_1": 18, "Plant_2": 16},
        "type_capacity": {
            "Plant_1": {"Small_Bioreactor": 14, "Large_Bioreactor": 7},
            "Plant_2": {"Small_Bioreactor": 12, "Large_Bioreactor": 6},
        },
        "manufacturing_cost": {
            "Plant_1": {"Small_Bioreactor": 105, "Large_Bioreactor": 205},
            "Plant_2": {"Small_Bioreactor": 112, "Large_Bioreactor": 198},
        },
        "setup_cost": {
            "Plant_1": {"Small_Bioreactor": 45, "Large_Bioreactor": 65},
            "Plant_2": {"Small_Bioreactor": 40, "Large_Bioreactor": 60},
        },
        "plant_holding_cost": {"Plant_1": 4, "Plant_2": 5},
        "dc_holding_cost": {"DC_1": 3, "DC_2": 3},
        "plant_to_dc_cost": {
            "Plant_1": {"DC_1": 8, "DC_2": 12},
            "Plant_2": {"DC_1": 11, "DC_2": 7},
        },
        "dc_to_market_cost": {
            "DC_1": {"Market_1": 4, "Market_2": 7},
            "DC_2": {"Market_1": 7, "Market_2": 4},
        },
        "distribution_lead_time": {
            "Plant_1": {"DC_1": 1, "DC_2": 1},
            "Plant_2": {"DC_1": 1, "DC_2": 1},
        },
        "initial_plant_inventory": {
            "Plant_1": {"Small_Bioreactor": 0, "Large_Bioreactor": 0},
            "Plant_2": {"Small_Bioreactor": 0, "Large_Bioreactor": 0},
        },
        "initial_dc_inventory": {
            "DC_1": {"Small_Bioreactor": 4, "Large_Bioreactor": 2},
            "DC_2": {"Small_Bioreactor": 3, "Large_Bioreactor": 1},
        },
        "plant_inventory_capacity": {"Plant_1": 30, "Plant_2": 30},
        "dc_inventory_capacity": {"DC_1": 30, "DC_2": 30},
    }


def robust_demand(data: dict[str, Any], t: int, market: str, product: str) -> int:
    """Return the upper endpoint of the box uncertainty set."""
    return (
        data["nominal_demand"][t][market][product]
        + data["demand_deviation"][t][market][product]
    )


def validate_data(data: dict[str, Any]) -> None:
    """Validate dimensions and non-negativity required by the robust model."""
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
        for market in data["markets"]:
            for product in data["products"]:
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
        required_first = sum(
            robust_demand(data, first, market, product) for market in data["markets"]
        )
        available_first = sum(
            data["initial_dc_inventory"][dc][product] for dc in data["dcs"]
        )
        if available_first < required_first:
            raise ValueError(
                f"Initial DC inventory for {product} is insufficient for robust first-period demand."
            )


def build_model(data: dict[str, Any]) -> RobustArtifacts:
    """Build a robust MILP with raw-material procurement and box demand uncertainty."""
    validate_data(data)
    periods = data["periods"]
    suppliers = data["suppliers"]
    plants = data["plants"]
    dcs = data["dcs"]
    markets = data["markets"]
    products = data["products"]
    materials = data["raw_materials"]

    model = pulp.LpProblem("Robust_Bioreactor_Supply_Chain_Optimization", pulp.LpMinimize)

    procurement = pulp.LpVariable.dicts(
        "Procurement",
        [(t, s, p, r) for t in periods for s in suppliers for p in plants for r in materials],
        lowBound=0,
        cat=pulp.LpInteger,
    )
    raw_inventory = pulp.LpVariable.dicts(
        "RawInventory",
        [(t, p, r) for t in periods for p in plants for r in materials],
        lowBound=0,
        cat=pulp.LpInteger,
    )
    production = pulp.LpVariable.dicts(
        "Production",
        [(t, p, k) for t in periods for p in plants for k in products],
        lowBound=0,
        cat=pulp.LpInteger,
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
        cat=pulp.LpInteger,
    )
    dc_inventory = pulp.LpVariable.dicts(
        "DCInventory",
        [(t, d, k) for t in periods for d in dcs for k in products],
        lowBound=0,
        cat=pulp.LpInteger,
    )
    plant_to_dc = pulp.LpVariable.dicts(
        "PlantToDC",
        [(t, p, d, k) for t in periods for p in plants for d in dcs for k in products],
        lowBound=0,
        cat=pulp.LpInteger,
    )
    dc_to_market = pulp.LpVariable.dicts(
        "DCToMarket",
        [(t, d, m, k) for t in periods for d in dcs for m in markets for k in products],
        lowBound=0,
        cat=pulp.LpInteger,
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

    for t in periods:
        for plant in plants:
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
                outbound = pulp.lpSum(dc_to_market[(t, dc, market, product)] for market in markets)
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
        for market in markets:
            for product in products:
                model += (
                    pulp.lpSum(dc_to_market[(t, dc, market, product)] for dc in dcs)
                    >= robust_demand(data, t, market, product),
                    f"RobustDemand_{t}_{market}_{product}",
                )

    last = periods[-1]
    for t in periods:
        for supplier in suppliers:
            for plant in plants:
                if t + data["procurement_lead_time"][supplier][plant] > last:
                    for material in materials:
                        model += (
                            procurement[(t, supplier, plant, material)] == 0,
                            f"NoLateProcurement_{t}_{supplier}_{plant}_{material}",
                        )
        for plant in plants:
            for dc in dcs:
                if t + data["distribution_lead_time"][plant][dc] > last:
                    for product in products:
                        model += (
                            plant_to_dc[(t, plant, dc, product)] == 0,
                            f"NoLateDistribution_{t}_{plant}_{dc}_{product}",
                        )

    return RobustArtifacts(
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
    """Solve the robust procurement model and return only an optimal solution."""
    artifacts = build_model(data)
    status_code = artifacts.model.solve(pulp.PULP_CBC_CMD(msg=solver_msg))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        raise RuntimeError(f"Optimization did not produce an optimal solution. Solver status: {status}")

    def values(variables: dict) -> dict:
        return {key: int(round(variable.value())) for key, variable in variables.items()}

    cost_breakdown = {
        name: float(pulp.value(expression)) for name, expression in artifacts.cost_terms.items()
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


def print_summary(results: dict[str, Any]) -> None:
    """Print a compact summary of the robust solution."""
    print("Robust Procurement Optimization Results")
    print("---------------------------------------")
    print(f"Status: {results['status']}")
    print(f"Total cost: {results['total_cost']:,.2f}")
    print(f"Total raw-material procurement: {sum(results['procurement'].values())} units")
    print(f"Total bioreactor production: {sum(results['production'].values())} units")
    print(f"Total market deliveries: {sum(results['dc_to_market'].values())} units")
    print("\nCost breakdown:")
    for name, value in results["cost_breakdown"].items():
        print(f"  {name}: {value:,.2f}")


def main() -> None:
    data = build_sample_data()
    results = solve_model(data)
    print_summary(results)


if __name__ == "__main__":
    main()
