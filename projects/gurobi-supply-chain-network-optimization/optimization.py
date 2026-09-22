import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def solve_supply_chain(
    suppliers,
    dcs,
    customers,
    carbon_price=0.02,
    min_open_dcs=4,
    max_open_dcs=7,
    preferred_service_radius_km=1400,
):
    s_names = suppliers["name"].tolist()
    d_names = dcs["name"].tolist()
    c_names = customers["name"].tolist()

    s_idx = suppliers.set_index("name")
    d_idx = dcs.set_index("name")
    c_idx = customers.set_index("name")

    sd_distance = {
        (s, d): haversine_km(
            s_idx.loc[s, "latitude"], s_idx.loc[s, "longitude"],
            d_idx.loc[d, "latitude"], d_idx.loc[d, "longitude"]
        )
        for s in s_names for d in d_names
    }
    dc_distance = {
        (d, c): haversine_km(
            d_idx.loc[d, "latitude"], d_idx.loc[d, "longitude"],
            c_idx.loc[c, "latitude"], c_idx.loc[c, "longitude"]
        )
        for d in d_names for c in c_names
    }

    # Hypothetical but internally consistent economics.
    # Dollar cost per unit-km and kg CO2 per unit-km.
    inbound_rate = 0.055
    outbound_rate = 0.075
    emission_factor = 0.00042
    long_haul_penalty_rate = 0.06

    model = gp.Model("SupplyChainNetworkOptimization")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.001

    x = model.addVars(s_names, d_names, lb=0.0, vtype=GRB.CONTINUOUS, name="supplier_to_dc")
    y = model.addVars(d_names, c_names, lb=0.0, vtype=GRB.CONTINUOUS, name="dc_to_customer")
    open_dc = model.addVars(d_names, vtype=GRB.BINARY, name="open_dc")

    # Exact customer assignment is allowed to split across DCs. This is a continuous network-flow model
    # with binary facility-location decisions.

    production_cost = gp.quicksum(
        float(s_idx.loc[s, "production_cost"]) * x[s, d]
        for s in s_names for d in d_names
    )
    inbound_transport = gp.quicksum(
        inbound_rate * sd_distance[s, d] * x[s, d]
        for s in s_names for d in d_names
    )
    outbound_transport = gp.quicksum(
        outbound_rate * dc_distance[d, c] * y[d, c]
        for d in d_names for c in c_names
    )
    facility_cost = gp.quicksum(
        float(d_idx.loc[d, "fixed_cost"]) * open_dc[d]
        for d in d_names
    )
    emissions_cost = carbon_price * gp.quicksum(
        emission_factor * sd_distance[s, d] * x[s, d]
        for s in s_names for d in d_names
    ) + carbon_price * gp.quicksum(
        emission_factor * dc_distance[d, c] * y[d, c]
        for d in d_names for c in c_names
    )

    # Soft service-distance penalty: routes beyond the preferred radius remain feasible,
    # but become less attractive. This makes the trade-off visible without forcing infeasibility.
    service_penalty = gp.quicksum(
        long_haul_penalty_rate * max(0.0, dc_distance[d, c] - preferred_service_radius_km) * y[d, c]
        for d in d_names for c in c_names
    )

    model.setObjective(
        production_cost + inbound_transport + outbound_transport + facility_cost + emissions_cost + service_penalty,
        GRB.MINIMIZE,
    )

    for s in s_names:
        model.addConstr(
            gp.quicksum(x[s, d] for d in d_names) <= float(s_idx.loc[s, "capacity"]),
            name=f"supplier_capacity[{s}]",
        )

    for c in c_names:
        model.addConstr(
            gp.quicksum(y[d, c] for d in d_names) == float(c_idx.loc[c, "demand"]),
            name=f"customer_demand[{c}]",
        )

    # Critical correction: exact flow conservation at every DC.
    for d in d_names:
        inbound = gp.quicksum(x[s, d] for s in s_names)
        outbound = gp.quicksum(y[d, c] for c in c_names)
        model.addConstr(inbound == outbound, name=f"flow_conservation[{d}]")
        model.addConstr(outbound <= float(d_idx.loc[d, "capacity"]) * open_dc[d], name=f"dc_capacity[{d}]")

    model.addConstr(gp.quicksum(open_dc[d] for d in d_names) >= min_open_dcs, name="min_open_dcs")
    model.addConstr(gp.quicksum(open_dc[d] for d in d_names) <= max_open_dcs, name="max_open_dcs")

    model.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
    }
    status = status_map.get(model.Status, str(model.Status))
    if model.Status != GRB.OPTIMAL:
        return {"status": status}

    flow_rows = []
    total_emissions_kg = 0.0

    for s in s_names:
        for d in d_names:
            q = x[s, d].X
            if q > 1e-6:
                dist = sd_distance[s, d]
                cost = inbound_rate * dist * q
                emissions = emission_factor * dist * q
                total_emissions_kg += emissions
                flow_rows.append({
                    "stage": "Supplier → DC",
                    "from": s,
                    "to": d,
                    "quantity": q,
                    "distance_km": dist,
                    "transport_cost": cost,
                })

    weighted_customer_distance = 0.0
    total_demand = float(customers["demand"].sum())
    for d in d_names:
        for c in c_names:
            q = y[d, c].X
            if q > 1e-6:
                dist = dc_distance[d, c]
                cost = outbound_rate * dist * q
                emissions = emission_factor * dist * q
                total_emissions_kg += emissions
                weighted_customer_distance += q * dist
                flow_rows.append({
                    "stage": "DC → Customer",
                    "from": d,
                    "to": c,
                    "quantity": q,
                    "distance_km": dist,
                    "transport_cost": cost,
                })

    dc_rows = []
    for d in d_names:
        throughput = sum(y[d, c].X for c in c_names)
        capacity = float(d_idx.loc[d, "capacity"])
        dc_rows.append({
            "distribution_center": d,
            "open": bool(open_dc[d].X > 0.5),
            "throughput": throughput,
            "capacity": capacity,
            "utilization": throughput / capacity if capacity else 0.0,
        })

    return {
        "status": status,
        "total_cost": model.ObjVal,
        "open_dc_count": sum(1 for d in d_names if open_dc[d].X > 0.5),
        "open_dcs": [d for d in d_names if open_dc[d].X > 0.5],
        "flows": pd.DataFrame(flow_rows),
        "dc_utilization": pd.DataFrame(dc_rows),
        "avg_customer_distance_km": weighted_customer_distance / total_demand,
        "total_emissions_kg": total_emissions_kg,
        "solver_runtime_sec": model.Runtime,
        "mip_gap": model.MIPGap,
    }
