from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pulp


@dataclass(frozen=True)
class ModelArtifacts:
    model: pulp.LpProblem
    production: dict
    setup: dict
    plant_inventory: dict
    dc_inventory: dict
    plant_to_dc: dict
    dc_to_market: dict
    cost_terms: dict[str, pulp.LpAffineExpression]


def build_sample_data() -> dict[str, Any]:
    """Return a deterministic, synthetic, multi-period planning instance."""
    periods = [1, 2, 3, 4]
    plants = ["Plant_1", "Plant_2"]
    dcs = ["DC_1", "DC_2"]
    markets = ["Market_1", "Market_2"]
    products = ["Small_Bioreactor", "Large_Bioreactor"]

    demand = {
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
            "Market_2": {"Small_Bioreactor": 4, "Large_Bioreactor": 2},
        },
        4: {
            "Market_1": {"Small_Bioreactor": 4, "Large_Bioreactor": 2},
            "Market_2": {"Small_Bioreactor": 3, "Large_Bioreactor": 1},
        },
    }

    return {
        "periods": periods,
        "plants": plants,
        "dcs": dcs,
        "markets": markets,
        "products": products,
        "demand": demand,
        "production_capacity": {"Plant_1": 16, "Plant_2": 14},
        "type_capacity": {
            "Plant_1": {"Small_Bioreactor": 12, "Large_Bioreactor": 6},
            "Plant_2": {"Small_Bioreactor": 10, "Large_Bioreactor": 5},
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
        "lead_time": {
            "Plant_1": {"DC_1": 1, "DC_2": 2},
            "Plant_2": {"DC_1": 1, "DC_2": 1},
        },
        "initial_plant_inventory": {
            "Plant_1": {"Small_Bioreactor": 0, "Large_Bioreactor": 0},
            "Plant_2": {"Small_Bioreactor": 0, "Large_Bioreactor": 0},
        },
        "initial_dc_inventory": {
            "DC_1": {"Small_Bioreactor": 3, "Large_Bioreactor": 1},
            "DC_2": {"Small_Bioreactor": 2, "Large_Bioreactor": 1},
        },
        "plant_inventory_capacity": {"Plant_1": 20, "Plant_2": 20},
        "dc_inventory_capacity": {"DC_1": 24, "DC_2": 24},
    }


def validate_data(data: dict[str, Any]) -> None:
    """Validate model data before any decision variables are created."""
    required = {
        "periods",
        "plants",
        "dcs",
        "markets",
        "products",
        "demand",
        "production_capacity",
        "type_capacity",
        "manufacturing_cost",
        "setup_cost",
        "plant_holding_cost",
        "dc_holding_cost",
        "plant_to_dc_cost",
        "dc_to_market_cost",
        "lead_time",
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
                if data["demand"][t][market][product] < 0:
                    raise ValueError("Demand must be non-negative.")

    for plant in data["plants"]:
        if data["production_capacity"][plant] < 0:
            raise ValueError("Production capacity must be non-negative.")
        for product in data["products"]:
            if data["type_capacity"][plant][product] < 0:
                raise ValueError("Type capacity must be non-negative.")

    for plant in data["plants"]:
        for dc in data["dcs"]:
            lead = data["lead_time"][plant][dc]
            if not isinstance(lead, int) or lead < 1:
                raise ValueError("Plant-to-DC lead times must be positive integers.")

    first_period = periods[0]
    for product in data["products"]:
        first_demand = sum(
            data["demand"][first_period][market][product]
            for market in data["markets"]
        )
        first_inventory = sum(
            data["initial_dc_inventory"][dc][product] for dc in data["dcs"]
        )
        if first_inventory < first_demand:
            raise ValueError(
                f"Initial DC inventory for {product} is insufficient for first-period demand."
            )


def build_model(data: dict[str, Any]) -> ModelArtifacts:
    """Build the mixed-integer linear programming model."""
    validate_data(data)

    periods = data["periods"]
    plants = data["plants"]
    dcs = data["dcs"]
    markets = data["markets"]
    products = data["products"]

    model = pulp.LpProblem("Bioreactor_Supply_Chain_Optimization", pulp.LpMinimize)

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
        "manufacturing": pulp.lpSum(
            data["manufacturing_cost"][p][k] * production[(t, p, k)]
            for t in periods
            for p in plants
            for k in products
        ),
        "setup": pulp.lpSum(
            data["setup_cost"][p][k] * setup[(t, p, k)]
            for t in periods
            for p in plants
            for k in products
        ),
        "plant_inventory": pulp.lpSum(
            data["plant_holding_cost"][p] * plant_inventory[(t, p, k)]
            for t in periods
            for p in plants
            for k in products
        ),
        "dc_inventory": pulp.lpSum(
            data["dc_holding_cost"][d] * dc_inventory[(t, d, k)]
            for t in periods
            for d in dcs
            for k in products
        ),
        "plant_to_dc_transport": pulp.lpSum(
            data["plant_to_dc_cost"][p][d] * plant_to_dc[(t, p, d, k)]
            for t in periods
            for p in plants
            for d in dcs
            for k in products
        ),
        "dc_to_market_transport": pulp.lpSum(
            data["dc_to_market_cost"][d][m] * dc_to_market[(t, d, m, k)]
            for t in periods
            for d in dcs
            for m in markets
            for k in products
        ),
    }
    model += pulp.lpSum(cost_terms.values()), "TotalCost"

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
                    <= data["type_capacity"][plant][product]
                    * setup[(t, plant, product)],
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
                outbound = pulp.lpSum(
                    plant_to_dc[(t, plant, dc, product)] for dc in dcs
                )
                model += (
                    plant_inventory[(t, plant, product)]
                    == previous + production[(t, plant, product)] - outbound,
                    f"PlantBalance_{t}_{plant}_{product}",
                )

            model += (
                pulp.lpSum(
                    plant_inventory[(t, plant, product)] for product in products
                )
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
                arriving = pulp.lpSum(
                    plant_to_dc[(depart_t, plant, dc, product)]
                    for plant in plants
                    for depart_t in periods
                    if depart_t + data["lead_time"][plant][dc] == t
                )
                outbound = pulp.lpSum(
                    dc_to_market[(t, dc, market, product)] for market in markets
                )
                model += (
                    dc_inventory[(t, dc, product)] == previous + arriving - outbound,
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
                    pulp.lpSum(
                        dc_to_market[(t, dc, market, product)] for dc in dcs
                    )
                    == data["demand"][t][market][product],
                    f"Demand_{t}_{market}_{product}",
                )

    last_period = periods[-1]
    for t in periods:
        for plant in plants:
            for dc in dcs:
                if t + data["lead_time"][plant][dc] > last_period:
                    for product in products:
                        model += (
                            plant_to_dc[(t, plant, dc, product)] == 0,
                            f"NoLateShipment_{t}_{plant}_{dc}_{product}",
                        )

    return ModelArtifacts(
        model=model,
        production=production,
        setup=setup,
        plant_inventory=plant_inventory,
        dc_inventory=dc_inventory,
        plant_to_dc=plant_to_dc,
        dc_to_market=dc_to_market,
        cost_terms=cost_terms,
    )


def solve_model(data: dict[str, Any], solver_msg: bool = False) -> dict[str, Any]:
    """Solve the model and reject non-optimal solver states."""
    artifacts = build_model(data)
    status_code = artifacts.model.solve(pulp.PULP_CBC_CMD(msg=solver_msg))
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        raise RuntimeError(
            f"Optimization did not produce an optimal solution. Solver status: {status}"
        )

    def values(variable_dict: dict) -> dict:
        return {
            key: int(round(variable.value()))
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
        "production": values(artifacts.production),
        "setup": values(artifacts.setup),
        "plant_inventory": values(artifacts.plant_inventory),
        "dc_inventory": values(artifacts.dc_inventory),
        "plant_to_dc": values(artifacts.plant_to_dc),
        "dc_to_market": values(artifacts.dc_to_market),
    }


def results_to_frames(results: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Convert all decision-variable values, including zeros, to DataFrames."""
    production = pd.DataFrame(
        [
            {"Period": t, "Plant": plant, "Product": product, "Production": value}
            for (t, plant, product), value in results["production"].items()
        ]
    )
    plant_inventory = pd.DataFrame(
        [
            {
                "Period": t,
                "Location": plant,
                "Product": product,
                "Inventory": value,
                "Echelon": "Plant",
            }
            for (t, plant, product), value in results["plant_inventory"].items()
        ]
    )
    dc_inventory = pd.DataFrame(
        [
            {
                "Period": t,
                "Location": dc,
                "Product": product,
                "Inventory": value,
                "Echelon": "Distribution Center",
            }
            for (t, dc, product), value in results["dc_inventory"].items()
        ]
    )
    plant_to_dc = pd.DataFrame(
        [
            {
                "Period": t,
                "From": plant,
                "To": dc,
                "Product": product,
                "Quantity": value,
            }
            for (t, plant, dc, product), value in results["plant_to_dc"].items()
        ]
    )
    dc_to_market = pd.DataFrame(
        [
            {
                "Period": t,
                "From": dc,
                "To": market,
                "Product": product,
                "Quantity": value,
            }
            for (t, dc, market, product), value in results["dc_to_market"].items()
        ]
    )
    return {
        "production": production,
        "inventory": pd.concat([plant_inventory, dc_inventory], ignore_index=True),
        "plant_to_dc": plant_to_dc,
        "dc_to_market": dc_to_market,
    }


def export_results(
    results: dict[str, Any], output_dir: str | Path = "results"
) -> dict[str, Path]:
    """Export decision variables and the cost decomposition to CSV files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = results_to_frames(results)

    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    cost_path = output / "cost_breakdown.csv"
    pd.DataFrame(
        [
            {"Cost Component": component, "Amount": amount}
            for component, amount in results["cost_breakdown"].items()
        ]
    ).to_csv(cost_path, index=False)
    paths["cost_breakdown"] = cost_path
    return paths


def plot_results(
    results: dict[str, Any], output_path: str | Path = "results/summary.png"
) -> Path:
    """Create a deterministic four-panel summary visualization."""
    frames = results_to_frames(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bioreactor Supply Chain Optimization Results", fontsize=16)

    production = frames["production"].pivot_table(
        index="Period",
        columns=["Plant", "Product"],
        values="Production",
        aggfunc="sum",
    ).fillna(0)
    production.plot(kind="bar", stacked=True, ax=axes[0, 0])
    axes[0, 0].set_title("Production by Plant and Product")
    axes[0, 0].set_ylabel("Units")

    inventory = frames["inventory"].pivot_table(
        index="Period", columns="Location", values="Inventory", aggfunc="sum"
    ).fillna(0)
    inventory.plot(marker="o", ax=axes[0, 1])
    axes[0, 1].set_title("Inventory by Location")
    axes[0, 1].set_ylabel("Units")

    market_flow = (
        frames["dc_to_market"]
        .groupby(["From", "To"], as_index=False)["Quantity"]
        .sum()
    )
    matrix = market_flow.pivot(index="From", columns="To", values="Quantity").fillna(0)
    image = axes[1, 0].imshow(matrix.values, aspect="auto")
    axes[1, 0].set_xticks(range(len(matrix.columns)), matrix.columns)
    axes[1, 0].set_yticks(range(len(matrix.index)), matrix.index)
    axes[1, 0].set_title("DC-to-Market Shipments")
    axes[1, 0].set_xlabel("Market")
    axes[1, 0].set_ylabel("Distribution Center")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 0].text(
                column,
                row,
                f"{matrix.iloc[row, column]:.0f}",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    labels = list(results["cost_breakdown"].keys())
    amounts = [results["cost_breakdown"][label] for label in labels]
    axes[1, 1].bar(labels, amounts)
    axes[1, 1].set_title("Actual Cost Breakdown")
    axes[1, 1].set_ylabel("Cost")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].text(
        0.5,
        0.95,
        f"Total Cost: {results['total_cost']:,.2f}",
        transform=axes[1, 1].transAxes,
        ha="center",
        va="top",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_summary(results: dict[str, Any]) -> None:
    """Print a compact solver and cost summary."""
    print("Optimization Results")
    print("--------------------")
    print(f"Status: {results['status']}")
    print(f"Total cost: {results['total_cost']:,.2f}")
    print(f"Total production: {sum(results['production'].values())} units")
    print(
        "Total plant-to-DC shipments: "
        f"{sum(results['plant_to_dc'].values())} units"
    )
    print(
        "Total DC-to-market shipments: "
        f"{sum(results['dc_to_market'].values())} units"
    )
    print("\nCost breakdown:")
    for name, value in results["cost_breakdown"].items():
        print(f"  {name}: {value:,.2f}")


def main() -> None:
    data = build_sample_data()
    results = solve_model(data)
    print_summary(results)
    export_results(results)
    plot_results(results)
    print("\nFiles written to the results/ directory.")


if __name__ == "__main__":
    main()
