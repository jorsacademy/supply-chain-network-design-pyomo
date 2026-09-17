# Supply-Chain and Inventory Optimization Research Series

This repository belongs to a broader set of independent projects on supply-chain network design, inventory control, uncertainty, and learning-based supply-chain decisions.

## Network design and deterministic planning

| Repository | Main focus | Role in the series |
|---|---|---|
| `supply-chain-network-design-pyomo` | Multi-echelon network design with facility-opening and flow decisions in Pyomo | Mathematical-programming foundation |
| `gurobi-supply-chain-network-optimization` | Capacitated facility location plus multi-echelon flow with Gurobi and interactive visualization | Applied Gurobi/network-flow project |
| `bioreactor-supply-chain-optimization-milp` | Supply-chain planning in a bioreactor/bioprocess context | Domain-specific MILP |
| `multimodal-distribution-network-genetic-algorithm` | Distribution-network design with a genetic algorithm | Metaheuristic network design |
| `multi-site-resource-allocation-optimization` | Resource allocation across multiple locations | Allocation companion |

## Inventory and demand planning

| Repository | Main focus | Role in the series |
|---|---|---|
| `seasonal-inventory-planning-python` | Seasonal inventory planning | Deterministic/seasonal inventory foundation |
| `multi-echelon-inventory-optimization` | Inventory decisions across multiple echelons | Multi-echelon inventory |
| `demand-forecasting-plus-inventory-control` | Forecasting connected to inventory decisions | Predict-then-control workflow |
| `chance-constrained-inventory-optimization-python` | Service/risk constraints under uncertain demand | Chance-constrained inventory |
| `wasserstein-dro-inventory-optimization-python` | Inventory decisions under distributional ambiguity | DRO inventory |
| `conformal-prediction-robust-inventory-optimization-python` | Calibrated predictive uncertainty converted into robust decisions | Conformal optimization |
| `robust-healthcare-inventory-optimization` | Robust inventory planning in healthcare | Domain-specific robust inventory |
| `pymdptoolbox-inventory-control` | Sequential inventory control as an MDP | Dynamic inventory control |
| `approximate-dynamic-programming-fleet-inventory` | Approximate dynamic programming for coupled fleet/inventory decisions | ADP extension |

## Robust, stochastic, and generative supply-chain planning

| Repository | Main focus | Role in the series |
|---|---|---|
| `robust-supply-chain-network-optimization` | Robust network design | Robust optimization |
| `distributionally-robust-supply-chain-optimization` | Distributionally robust supply-chain planning | DRO extension |
| `generative-supply-chain-scenarios-stochastic-optimization-pytorch` | Learned scenario generation feeding stochastic optimization | Generative uncertainty modeling |
| `two-stage-stochastic-capacity-planning` | Two-stage planning under uncertainty | Stochastic-programming foundation |
| `mpi-sppy-multistage-stochastic-planning` | Distributed/multistage stochastic programming | Multistage decomposition/tooling |

## Sequential and learning-based supply-chain decisions

- `hierarchical-supply-chain-rl` — hierarchical reinforcement learning for multi-level supply-chain decisions.
- `contextual-bandits-dynamic-procurement` — adaptive procurement with contextual bandits.
- `bilevel-supply-chain-pricing-optimization` — strategic pricing/leader-follower optimization.
- `ml-assisted-assortment-optimization` — machine-learning-assisted assortment decisions.

## Why these repositories remain separate

Network design, inventory control, robust optimization, distributionally robust optimization, stochastic programming, MDP/ADP control, contextual bandits, bilevel pricing, and hierarchical RL impose different information structures and solution methods. They form one decision domain but not one algorithmic project.

The Pyomo and Gurobi network-design repositories also remain separate: one emphasizes a multi-echelon mathematical formulation translated from GAMS/Pyomo, while the other combines facility location, network flow, geographic costs, visualization, and scenario interaction.

The ordering of projects in this document is organizational and pedagogical, not a ranking.