import folium


def _lookup(df):
    return df.set_index("name")[["latitude", "longitude"]].to_dict("index")


def build_network_map(suppliers, dcs, customers, solution):
    supplier_pos = _lookup(suppliers)
    dc_pos = _lookup(dcs)
    customer_pos = _lookup(customers)
    open_dcs = set(solution["open_dcs"])
    flows = solution["flows"]

    m = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Draw optimized lanes first so markers remain visible.
    if not flows.empty:
        max_q = max(float(flows["quantity"].max()), 1.0)
        for _, row in flows.iterrows():
            origin = row["from"]
            destination = row["to"]
            q = float(row["quantity"])
            width = 1.5 + 7.0 * (q / max_q) ** 0.65

            if row["stage"] == "Supplier → DC":
                p1 = supplier_pos[origin]
                p2 = dc_pos[destination]
                color = "#2563eb"
                dash = "5,7"
            else:
                p1 = dc_pos[origin]
                p2 = customer_pos[destination]
                color = "#16a34a"
                dash = None

            tooltip = (
                f"{origin} → {destination}<br>"
                f"Flow: {q:,.0f} units<br>"
                f"Distance: {float(row['distance_km']):,.0f} km"
            )
            folium.PolyLine(
                [(p1["latitude"], p1["longitude"]), (p2["latitude"], p2["longitude"])],
                color=color,
                weight=width,
                opacity=0.72,
                dash_array=dash,
                tooltip=tooltip,
            ).add_to(m)

    for _, row in suppliers.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color="#1d4ed8",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.95,
            weight=2,
            tooltip=f"{row['name']} | Capacity {row['capacity']:,.0f}",
        ).add_to(m)

    util = solution["dc_utilization"].set_index("distribution_center")
    for _, row in dcs.iterrows():
        is_open = row["name"] in open_dcs
        utilization = float(util.loc[row["name"], "utilization"]) if row["name"] in util.index else 0.0
        color = "#15803d" if is_open else "#9ca3af"
        fill = "#22c55e" if is_open else "#d1d5db"
        status = "OPEN" if is_open else "NOT SELECTED"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=10 if is_open else 7,
            color=color,
            fill=True,
            fill_color=fill,
            fill_opacity=0.95 if is_open else 0.65,
            weight=3 if is_open else 1,
            tooltip=(
                f"{row['name']} | {status}<br>"
                f"Capacity: {row['capacity']:,.0f}<br>"
                f"Utilization: {utilization:.0%}"
            ),
        ).add_to(m)

    for _, row in customers.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5 + min(float(row["demand"]) / 90.0, 4.5),
            color="#b91c1c",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.9,
            weight=1.5,
            tooltip=f"{row['name']} | Demand {row['demand']:,.0f}",
        ).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index:9999;
                background:white; padding:10px 12px; border:1px solid #d1d5db;
                border-radius:8px; font-size:13px; box-shadow:0 1px 5px rgba(0,0,0,.15)">
      <b>Network legend</b><br>
      <span style="color:#3b82f6">●</span> Supplier<br>
      <span style="color:#22c55e">●</span> Open DC<br>
      <span style="color:#9ca3af">●</span> Candidate DC not selected<br>
      <span style="color:#ef4444">●</span> Customer market<br>
      <span style="color:#2563eb">━━</span> Supplier → DC flow<br>
      <span style="color:#16a34a">━━</span> DC → Customer flow
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m
