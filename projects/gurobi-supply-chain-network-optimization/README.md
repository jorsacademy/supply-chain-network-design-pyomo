# Gurobi Supply Chain Network Optimization

An applied Operations Research project that combines **capacitated facility location** and **multi-echelon network flow** optimization, solved with **Gurobi** and presented interactively with **Streamlit** and **Folium**.

The dataset is hypothetical but geographically plausible. The model decides which candidate distribution centers (DCs) to open and how product should flow from suppliers to DCs and from DCs to customer markets.

## Why this is more than a simple transportation model

The optimization includes:

- 8 suppliers across the United States
- 12 candidate distribution centers
- 24 customer markets
- Binary DC-opening decisions
- Supplier production capacities
- DC throughput capacities
- Exact customer-demand satisfaction
- Exact flow conservation at every DC
- Great-circle (Haversine) distances
- Inbound and outbound transportation costs
- Fixed facility costs
- Optional CO₂ penalty
- Soft penalty for very long customer-service lanes
- Minimum and maximum number of open DCs

Candidate DCs that are not selected remain visible on the map in gray. This is intentional: they represent alternatives the optimization rejected.

## Mathematical formulation

Sets:

- `S`: suppliers
- `D`: candidate distribution centers
- `C`: customer markets

Decision variables:

- `x[s,d] >= 0`: units shipped from supplier `s` to DC `d`
- `y[d,c] >= 0`: units shipped from DC `d` to customer `c`
- `z[d] ∈ {0,1}`: 1 if DC `d` is opened

The objective minimizes production, inbound transport, outbound transport, fixed DC-opening, carbon, and long-haul service penalty costs.

Key constraints:

1. Supplier capacity
2. Exact customer-demand satisfaction
3. **Flow conservation at every DC:** inbound flow equals outbound flow
4. DC throughput capacity linked to the binary opening decision
5. Bounds on the number of DCs that may be opened

The flow-conservation constraint is important. It prevents the physically invalid result where a DC could ship goods to customers without receiving them from suppliers.

## Streamlit output

The app displays:

- total optimized cost
- number of selected DCs
- weighted average customer-service distance
- estimated CO₂ emissions
- one interactive US network map
- supplier → DC lanes
- DC → customer lanes
- line widths proportional to shipment volume
- open versus rejected candidate DCs
- optimized flow table
- DC utilization table
- interactive scenario controls

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

A working Gurobi installation and license are required. Academic users can obtain licensing directly from Gurobi under Gurobi's own terms.

## Run

```bash
streamlit run app.py
```

## Project structure

```text
.
├── app.py
├── data.py
├── optimization.py
├── visualization.py
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Data disclaimer

All supplier capacities, facility costs, demands, rates, and network assumptions in this repository are hypothetical and are intended for teaching and demonstration. Geographic coordinates correspond approximately to real US metropolitan areas, but the network does not represent a real company.

## License

This project is released under a custom **Non-Commercial Academic License**. Academic, educational, teaching, and non-commercial research use is permitted. Commercial use is prohibited without prior written permission.

This is **not** an OSI-approved open-source license. Third-party dependencies, including Gurobi, remain subject to their own licenses.
