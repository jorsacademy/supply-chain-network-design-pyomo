# Bilevel Supply-Chain Pricing Optimization

A transparent implementation of an **optimistic bilevel Stackelberg pricing problem** in which a manufacturer sets wholesale prices and a retailer responds by solving a continuous procurement LP.

The repository exposes the full modeling chain:

```text
manufacturer chooses wholesale prices
              |
              v
retailer solves a procurement LP
              |
              v
purchase quantities
              |
              v
manufacturer profit
```

The lower-level LP is replaced by its KKT conditions. Complementarity is converted into a mixed-integer linear formulation using Fortuny-Amat-style disjunctions with **problem-derived finite bounds**, not arbitrary trial-and-error constants. The resulting MILP is independently verified against an exact small-instance optimistic bilevel oracle based on follower extreme-point enumeration.

## Stackelberg model

For product `j`, let `r_j` be retail price, `c_j` manufacturer unit cost, `a_j` retailer resource use, `u_j` a purchase cap, and `w_j` the wholesale price. The retailer has shared capacity `B`.

The retailer solves:

```text
maximize      sum_j (r_j - w_j) q_j
subject to    sum_j a_j q_j <= B
              0 <= q_j <= u_j
```

The manufacturer solves:

```text
maximize      sum_j (w_j - c_j) q_j
subject to    w_lower_j <= w_j <= w_upper_j
              q is an optimal solution of the retailer LP
```

The manufacturer cannot directly choose `q`; it can only influence the retailer through wholesale prices.

## Optimistic bilevel semantics

If the retailer has multiple optimal responses at the same wholesale prices, this repository uses the **optimistic** convention: among follower-optimal responses, the one most favorable to the leader may be selected. A pessimistic formulation would protect against the least favorable follower-optimal response and is a different problem.

## KKT reformulation

Write the follower as a minimization problem and associate nonnegative multipliers `lambda`, `mu_j`, and `nu_j` with shared capacity, upper quantity bounds, and lower quantity bounds.

Stationarity is:

```text
w_j - r_j + a_j lambda + mu_j - nu_j = 0
```

Complementarity is:

```text
lambda * (B - a^T q) = 0
mu_j * (u_j - q_j) = 0
nu_j * q_j = 0
```

Because the follower is a feasible bounded LP, the KKT conditions are necessary and sufficient for lower-level optimality.

## Linearizing the leader objective

The original leader objective contains `w_j q_j`. Follower strong duality gives:

```text
sum_j (r_j - w_j) q_j
= B lambda + sum_j u_j mu_j
```

Therefore manufacturer profit can be written linearly as:

```text
sum_j (r_j - c_j) q_j
- B lambda
- sum_j u_j mu_j
```

The single-level reformulation is therefore a MILP.

## Complementarity bounds are derived, not guessed

A Fortuny-Amat linearization requires finite upper bounds on primal slacks and dual multipliers. Poorly chosen big-M values can invalidate a bilevel reformulation, so the constants here are derived from this restricted one-resource follower structure.

Let:

```text
m_max_j = r_j - w_lower_j
m_min_j = r_j - w_upper_j
```

A dual-optimal capacity multiplier can be chosen with:

```text
0 <= lambda <= max_j max(0, m_max_j) / a_j
```

and the bound multipliers satisfy valid structure-specific bounds:

```text
0 <= mu_j <= max(0, m_max_j)
0 <= nu_j <= max(0, a_j lambda_max - m_min_j)
```

Primal slack bounds follow directly from `B` and `u_j`. These formulas are not claimed to apply to arbitrary bilevel models.

## Independent exact oracle

The KKT MILP is not trusted merely because it solves successfully. For small instances, the repository independently enumerates extreme points of the retailer feasible polytope:

```text
P = {q : a^T q <= B, 0 <= q <= u}
```

For a candidate extreme point `q^k` to be retailer-optimal, it must dominate every other follower extreme point `q^l`:

```text
(r - w)^T q^k >= (r - w)^T q^l
```

which is equivalent to linear inequalities in `w`:

```text
(q^k - q^l)^T w <= (q^k - q^l)^T r
```

For each follower extreme point, the leader solves an LP over wholesale prices. The best feasible pair is the exact optimistic bilevel solution for this small model. This oracle is exponential and exists only for verification.

## Default result

The reproducible three-product instance uses:

```text
retail prices        [18, 15, 13]
production costs     [ 5,  4,  3]
resource use         [ 3,  2,  1.5]
quantity caps        [ 5,  8, 10]
retailer capacity     27
wholesale lower      [ 6,  5,  4]
wholesale upper      [16, 13, 11]
```

Run:

```bash
python -m bilevel_pricing
```

The KKT MILP and exact vertex oracle both return:

```text
manufacturer profit     134.0
retailer profit           32.0
quantities              [0, 6, 10]
```

One KKT-optimal wholesale-price vector is `[15, 13, 11]`. The exact oracle may return a different price for a zero-quantity product because that price does not affect either player's objective. The manufacturer objective values agree to numerical precision.

The benchmark also solves the vertically integrated channel LP and reports channel efficiency as a diagnostic benchmark.

## Verification

The test suite checks:

- data validation;
- follower LP feasibility;
- hand-checkable one-product cases;
- follower optimality of the KKT quantity vector;
- containment of KKT duals within derived complementarity bounds;
- feasibility and uniqueness of enumerated follower extreme points;
- exact agreement between KKT MILP and independent bilevel enumeration on the demo;
- exact agreement on eight random three-product instances;
- centralized channel benchmark consistency;
- CLI JSON output.

Local development result:

```text
18 passed
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11+ is required.

## Tests

```bash
pytest -q
```

GitHub Actions runs installation, compilation, Ruff, the full regression suite, and an end-to-end bilevel smoke test on Python 3.11 and 3.12.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_demo.py
├── src/bilevel_pricing/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── bounds.py
│   ├── exact.py
│   ├── follower.py
│   ├── kkt.py
│   └── model.py
├── tests/
│   ├── test_benchmark_cli.py
│   ├── test_bounds_kkt.py
│   ├── test_exact.py
│   └── test_model_follower.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Methodological boundaries

This repository does **not** claim to provide a general-purpose bilevel solver, pessimistic bilevel optimization, nonlinear or integer follower decisions, multiple-follower equilibrium modeling, general automatic big-M derivation, calibrated supply-chain demand, or production-solver speedups.

The follower is deliberately simple enough that the complementarity bounds can be derived analytically and the complete bilevel result can be independently verified.

Natural extensions include multiple retailer resources, network pricing, endogenous demand, pessimistic tie-breaking, multiple followers, and decomposition methods for larger instances.

## Research grounding

- J. Fortuny-Amat and B. McCarl, **A Representation and Economic Interpretation of a Two-Level Programming Problem**, *Journal of the Operational Research Society* 32(9), 783-792 (1981). DOI: `10.1057/jors.1981.156`.
- S. Dempe, **Foundations of Bilevel Programming**, Kluwer Academic Publishers, 2002.

Fortuny-Amat and McCarl developed the classical route of replacing a lower-level problem by Kuhn-Tucker conditions and exploiting the disjunctive structure of complementary slackness. Modern bilevel literature distinguishes optimistic and pessimistic handling of nonunique follower optima and emphasizes that invalid big-M values can change the bilevel optimum.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
