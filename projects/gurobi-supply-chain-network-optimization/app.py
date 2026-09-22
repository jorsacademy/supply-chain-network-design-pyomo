import math
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from data import build_demo_data
from optimization import solve_supply_chain
from visualization import build_network_map

st.set_page_config(page_title="Supply Chain Network Optimization", layout="wide")

st.title("Supply Chain Network Optimization")
st.caption("Capacitated facility-location and network-flow optimization with Gurobi")

suppliers, dcs, customers = build_demo_data()

with st.sidebar:
    st.header("Scenario controls")
    carbon_price = st.slider("Carbon penalty ($ / kg CO₂)", 0.0, 0.20, 0.02, 0.01)
    min_open_dcs = st.slider("Minimum open DCs", 3, len(dcs), 4)
    max_open_dcs = st.slider("Maximum open DCs", min_open_dcs, len(dcs), 7)
    service_radius_km = st.slider("Preferred service radius (km)", 500, 2500, 1400, 100)
    optimize = st.button("Optimize network", type="primary", use_container_width=True)

if "solution" not in st.session_state or optimize:
    st.session_state.solution = solve_supply_chain(
        suppliers=suppliers,
        dcs=dcs,
        customers=customers,
        carbon_price=carbon_price,
        min_open_dcs=min_open_dcs,
        max_open_dcs=max_open_dcs,
        preferred_service_radius_km=service_radius_km,
    )

solution = st.session_state.solution

if solution["status"] != "OPTIMAL":
    st.error(f"Optimization status: {solution['status']}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total cost", f"${solution['total_cost']:,.0f}")
m2.metric("Open DCs", f"{solution['open_dc_count']} / {len(dcs)}")
m3.metric("Average customer distance", f"{solution['avg_customer_distance_km']:,.0f} km")
m4.metric("CO₂ emissions", f"{solution['total_emissions_kg']:,.0f} kg")

st.subheader("Optimized Supply Chain Network")
st.caption("Blue = suppliers, green = open DCs, gray = candidate DCs not selected, red = customer markets. Line width is proportional to optimized shipment volume.")
network_map = build_network_map(suppliers, dcs, customers, solution)
st_folium(network_map, width=None, height=720, returned_objects=[])

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Optimized flows")
    flows = solution["flows"].copy()
    st.dataframe(
        flows.sort_values("quantity", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "quantity": st.column_config.NumberColumn("Units", format="%.0f"),
            "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.0f"),
            "transport_cost": st.column_config.NumberColumn("Transport cost", format="$%.0f"),
        },
    )

with right:
    st.subheader("Distribution center utilization")
    util = solution["dc_utilization"].copy()
    st.dataframe(
        util,
        use_container_width=True,
        hide_index=True,
        column_config={
            "open": st.column_config.CheckboxColumn("Open"),
            "throughput": st.column_config.NumberColumn("Throughput", format="%.0f"),
            "capacity": st.column_config.NumberColumn("Capacity", format="%.0f"),
            "utilization": st.column_config.ProgressColumn("Utilization", min_value=0.0, max_value=1.0, format="%.0%%"),
        },
    )

st.subheader("Model inputs")
t1, t2, t3 = st.tabs(["Suppliers", "Candidate DCs", "Customer markets"])
with t1:
    st.dataframe(suppliers, use_container_width=True, hide_index=True)
with t2:
    st.dataframe(dcs, use_container_width=True, hide_index=True)
with t3:
    st.dataframe(customers, use_container_width=True, hide_index=True)

st.subheader("What is being optimized?")
st.markdown(
    """
The model chooses which candidate distribution centers to open and how much product to ship on every Supplier→DC and DC→Customer lane. It minimizes fixed facility costs, transportation costs, and an optional carbon penalty while enforcing supplier capacities, DC throughput capacities, exact customer-demand satisfaction, and flow conservation at every open DC.
"""
)
