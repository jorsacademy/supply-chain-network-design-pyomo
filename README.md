# Supply Chain Network Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`bilevel-supply-chain-pricing-optimization`](projects/bilevel-supply-chain-pricing-optimization/)
- [`bioreactor-supply-chain-optimization-milp`](projects/bioreactor-supply-chain-optimization-milp/)
- [`gurobi-supply-chain-network-optimization`](projects/gurobi-supply-chain-network-optimization/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

This repository contains a mixed-integer linear programming (MILP) model for a multi-echelon supply chain network. The model is a Python/Pyomo translation of the original GAMS formulation.

The network contains:

- suppliers,
- factories,
- components,
- distribution centers,
- customers.

The optimization decides:

- how much of each component each supplier sends to each factory,
- how much final product each factory sends to each distribution center,
- how much each distribution center sends to each customer,
- which factories are opened,
- which distribution centers are opened.

The objective is to minimize total transportation and facility-opening cost while satisfying supplier capacities, factory capacities, distribution-center capacities, component requirements, flow conservation, and customer demand.

## Mathematical Structure

Decision variables:

- `X[i,j,t]`: quantity of component `t` shipped from supplier `i` to factory `j`
- `Y[j,k]`: quantity of final product shipped from factory `j` to distribution center `k`
- `Z[k,l]`: quantity shipped from distribution center `k` to customer `l`
- `Q[j]`: binary variable equal to 1 if factory `j` is opened
- `V[k]`: binary variable equal to 1 if distribution center `k` is opened

The bill of materials is:

- component 1: 3 units per finished product
- component 2: 1 unit per finished product
- component 3: 2 units per finished product

## Installation

```bash
pip install -r requirements.txt
```

The default solver is HiGHS, installed through `highspy`.

## Run

```bash
python supply_chain_model.py
```

The script prints the minimum total cost, opened facilities, and all nonzero shipment decisions.

## Model Fidelity

The Python implementation preserves the structure of the supplied GAMS model, including:

- supplier-component capacity constraints,
- factory-capacity constraints linked to binary opening decisions,
- a limit on the number of factories that may be opened,
- distribution-center-capacity constraints linked to binary opening decisions,
- a limit on the number of distribution centers that may be opened,
- component-balance equations at each factory,
- flow conservation at each distribution center,
- customer demand constraints,
- fixed factory and distribution-center opening costs.

## License

This project is released under the **JORS Academy Non-Commercial License 1.0**.

Commercial use is prohibited without a separate written commercial license from the copyright holder. This repository is therefore source-available for non-commercial use and is not distributed under an OSI-approved open-source license.

See [LICENSE](LICENSE) for the full terms.
