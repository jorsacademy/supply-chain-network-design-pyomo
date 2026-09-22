# Bioreactor Supply Chain Optimization MILP

This repository contains mixed-integer linear programming models for multi-period bioreactor supply-chain planning.

All entities, costs, capacities, and demand values are synthetic. No real or disguised company names, brands, or proprietary operating data are used.

The repository contains three models:

1. A deterministic production and distribution model.
2. A robust procurement model with raw-material sourcing and box demand uncertainty.
3. A Bertsimas-Sim budgeted robust procurement model with an adjustable uncertainty budget.

## Network Structures

Deterministic model:

`Plants -> Distribution Centers -> Markets`

Procurement models:

`Suppliers -> Plants -> Distribution Centers -> Markets`

## Scope

This project is an Operations Research example intended for educational, research, and other non-commercial use.

Commercial use is prohibited under the repository license.

## Deterministic Model

The deterministic implementation is located in:

```text
bioreactor_supply_chain.py
```

It jointly optimizes product-specific production quantities, binary setup decisions, plant inventory, distribution-center inventory, plant-to-DC shipments, and DC-to-market shipments.

The formulation includes production limits, inventory capacities, exact demand satisfaction, plant-to-DC transportation lead times, and protection against shipments that would arrive after the planning horizon.

## Box-Robust Procurement Model

The upstream procurement implementation is located in:

```text
robust_procurement_model.py
```

This model adds a complete upstream layer:

`Suppliers -> Plants`

and explicitly represents raw materials consumed during bioreactor production.

### Additional Decision Variables

- `Procurement[t, s, p, r]`: raw material `r` ordered from supplier `s` for plant `p`
- `RawInventory[t, p, r]`: end-of-period raw-material inventory at plant `p`

The model includes supplier capacity, supplier-to-plant lead times, raw-material inventory conservation, raw-material storage capacity, bills of materials, and production-linked material consumption.

The box-robust formulation protects every demand component at its maximum specified deviation:

```text
Delivered[t, m, k] >= NominalDemand[t, m, k] + DemandDeviation[t, m, k]
```

This is intentionally conservative because all demand deviations are protected simultaneously.

## Bertsimas-Sim Budgeted Robust Procurement Model

The budgeted robust implementation is located in:

```text
budgeted_robust_procurement_model.py
```

It uses the same four-echelon network and raw-material procurement logic, but replaces simultaneous full box protection with a Bertsimas-Sim uncertainty budget.

For each period `t` and product `k`, the user specifies:

```text
Gamma[t, k]
```

where:

```text
0 <= Gamma[t, k] <= number of markets
```

The demand deviations for that period-product pair are sorted from largest to smallest. If:

```text
Gamma = q + alpha
```

with integer `q = floor(Gamma)` and `0 <= alpha < 1`, the protection term is:

```text
Protection = sum(q largest deviations) + alpha * next-largest deviation
```

The aggregate protected demand is therefore:

```text
ProtectedTotalDemand[t, k]
= SumMarketNominalDemand[t, k]
+ Protection[t, k]
```

Every market must still receive at least its nominal demand:

```text
Delivered[t, m, k] >= NominalDemand[t, m, k]
```

and aggregate deliveries across markets must equal the protected total:

```text
sum_m Delivered[t, m, k] = ProtectedTotalDemand[t, k]
```

This formulation interprets the Bertsimas-Sim budget across market-level demand deviations for each period-product pair. It protects aggregate product demand while preventing any market from receiving less than its nominal requirement.

### Interpretation of Gamma

- `Gamma = 0`: nominal demand only
- `Gamma = 1`: protection against the largest market deviation
- `Gamma = number of markets`: equivalent aggregate protection to the full box upper bound
- Fractional `Gamma`: partial protection against the next-largest deviation

This provides a tunable conservatism parameter rather than assuming every market reaches its maximum deviation simultaneously.

## Raw-Material Logic

For each plant and raw material:

```text
Ending Raw Inventory
= Previous Raw Inventory
+ Arriving Procurement
- Production Consumption
```

Production consumption is calculated from the bill of materials:

```text
Consumption[p, r, t]
= sum(BOM[k, r] * Production[t, p, k])
```

Procurement orders become available only after the supplier-to-plant lead time.

## Objective Function

The deterministic model minimizes:

- Manufacturing cost
- Production setup cost
- Plant inventory holding cost
- Distribution-center inventory holding cost
- Plant-to-DC transportation cost
- DC-to-market transportation cost

The procurement models additionally include:

- Raw-material procurement cost
- Supplier-to-plant transportation cost
- Raw-material inventory holding cost

All reported cost components are calculated directly from the solved objective expression. No placeholder percentages are used.

## Lead-Time Logic

A procurement order dispatched from supplier `s` to plant `p` in period `tau` becomes available in:

```text
tau + procurement_lead_time[s][p]
```

A finished-product shipment dispatched from plant `p` to DC `d` in period `tau` becomes available in:

```text
tau + distribution_lead_time[p][d]
```

Late procurement and distribution decisions that would arrive after the planning horizon are forced to zero.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

The models use PuLP with the CBC MILP solver. Many PuLP installations include CBC. If CBC is unavailable on the host system, install the Coin-OR CBC solver separately.

## Run the Models

Deterministic model:

```bash
python bioreactor_supply_chain.py
```

Box-robust procurement model:

```bash
python robust_procurement_model.py
```

Bertsimas-Sim budgeted robust procurement model:

```bash
python budgeted_robust_procurement_model.py
```

## Tests

Run the complete regression suite with:

```bash
pytest -q
```

The test suite covers:

- Optimal solver status
- Exact deterministic demand satisfaction
- Distribution-center inventory balances with lead times
- Raw-material inventory conservation
- Procurement lead-time accounting
- Bill-of-material consumption
- Objective decomposition
- Box-robust demand protection
- Bertsimas-Sim protection calculations
- Fractional uncertainty budgets
- Nominal per-market service under budgeted robustness
- Aggregate protected-demand equality
- Invalid uncertainty-budget rejection
- Input validation

## Comparing Robustness Levels

The three formulations provide a natural conservatism ladder:

```text
Deterministic
    < Budgeted Robust with small Gamma
    < Budgeted Robust with large Gamma
    <= Full Box Robust
```

The budgeted robust formulation is useful when it is implausible that all market demand deviations reach their maximum simultaneously.

## Model Design Choices

Direct plant-to-market shipment is intentionally excluded. The downstream network therefore remains a strict two-echelon distribution structure.

Inter-DC shipment is also excluded. It can be introduced as a separate transshipment decision class if required.

The Bertsimas-Sim extension applies the uncertainty budget across market demand deviations for each product and period. It is an aggregate-demand protection model, not a probabilistic model and not a scenario-based stochastic program.

## Future Extensions

Natural extensions include:

- Two-stage stochastic programming with scenario probabilities
- Scenario-dependent recourse decisions
- Chance constraints
- Backorders and lost sales
- Service-level constraints
- Safety stock
- Supplier minimum-order quantities
- Supplier fixed-order costs
- Supplier disruption scenarios
- Multi-source qualification constraints
- Yield uncertainty
- Capacity expansion
- Overtime
- Carbon-emission objectives or constraints
- Multi-objective optimization

## Reproducibility

The included examples are deterministic and synthetic. No random demand, random transportation costs, external APIs, proprietary datasets, or hidden state are required.

## License

This repository is provided under a non-commercial software license. Commercial use is prohibited.

See [LICENSE](LICENSE) for the complete terms.

## Disclaimer

This project is a synthetic Operations Research example. It is not production-planning advice, regulatory guidance, procurement advice, or a validated model for any specific manufacturing operation.
