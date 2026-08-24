from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    NonNegativeReals, Binary, minimize, SolverFactory, value
)


def build_model():
    """Build the supply chain network design model translated from the GAMS formulation."""
    model = ConcreteModel(name="Supply Chain Network Design")

    # Sets
    model.suppliers = Set(initialize=[1, 2])
    model.factories = Set(initialize=[1, 2, 3, 4])
    model.components = Set(initialize=[1, 2, 3])
    model.distribution_centers = Set(initialize=[1, 2, 3, 4])
    model.customers = Set(initialize=[1, 2, 3, 4, 5])

    # Bill-of-materials coefficients: units of each component required per unit of final product
    W = {1: 3, 2: 1, 3: 2}

    # Supplier -> Factory unit transportation cost, indexed by (supplier, factory, component)
    C1 = {
        (1, 1, 1): 6, (1, 1, 2): 5, (1, 1, 3): 4,
        (1, 2, 1): 8, (1, 2, 2): 8, (1, 2, 3): 6,
        (1, 3, 1): 6, (1, 3, 2): 6, (1, 3, 3): 6,
        (1, 4, 1): 9, (1, 4, 2): 5, (1, 4, 3): 6,
        (2, 1, 1): 5, (2, 1, 2): 6, (2, 1, 3): 9,
        (2, 2, 1): 7, (2, 2, 2): 4, (2, 2, 3): 9,
        (2, 3, 1): 7, (2, 3, 2): 9, (2, 3, 3): 9,
        (2, 4, 1): 4, (2, 4, 2): 6, (2, 4, 3): 5,
    }

    # Factory -> Distribution Center unit transportation cost, indexed by (factory, dc)
    C2 = {
        (1, 1): 8, (1, 2): 7, (1, 3): 7, (1, 4): 7,
        (2, 1): 7, (2, 2): 5, (2, 3): 4, (2, 4): 5,
        (3, 1): 8, (3, 2): 4, (3, 3): 6, (3, 4): 6,
        (4, 1): 6, (4, 2): 5, (4, 3): 5, (4, 4): 5,
    }

    # Distribution Center -> Customer unit transportation cost, indexed by (dc, customer)
    C3 = {
        (1, 1): 5, (1, 2): 6, (1, 3): 5, (1, 4): 7, (1, 5): 5,
        (2, 1): 8, (2, 2): 4, (2, 3): 6, (2, 4): 5, (2, 5): 5,
        (3, 1): 6, (3, 2): 6, (3, 3): 6, (3, 4): 4, (3, 5): 4,
        (4, 1): 5, (4, 2): 8, (4, 3): 8, (4, 4): 7, (4, 5): 8,
    }

    # Factory capacities and fixed opening costs
    b = {1: 220, 2: 180, 3: 230, 4: 170}
    O = {1: 500, 2: 450, 3: 510, 4: 300}

    # Distribution center capacities and fixed opening costs
    c = {1: 170, 2: 230, 3: 310, 4: 210}
    S = {1: 400, 2: 460, 3: 610, 4: 450}

    # Customer demand
    d = {1: 100, 2: 80, 3: 90, 4: 120, 5: 110}

    # Supplier component capacities, indexed by (supplier, component)
    a = {
        (1, 1): 1350, (1, 2): 280, (1, 3): 710,
        (2, 1): 980, (2, 2): 610, (2, 3): 560,
    }

    model.W = Param(model.components, initialize=W)
    model.C1 = Param(model.suppliers, model.factories, model.components, initialize=C1)
    model.C2 = Param(model.factories, model.distribution_centers, initialize=C2)
    model.C3 = Param(model.distribution_centers, model.customers, initialize=C3)
    model.b = Param(model.factories, initialize=b)
    model.O = Param(model.factories, initialize=O)
    model.c = Param(model.distribution_centers, initialize=c)
    model.S = Param(model.distribution_centers, initialize=S)
    model.d = Param(model.customers, initialize=d)
    model.a = Param(model.suppliers, model.components, initialize=a)

    # Decision variables
    model.X = Var(
        model.suppliers, model.factories, model.components,
        domain=NonNegativeReals,
        doc="Component flow from supplier i to factory j"
    )
    model.Y = Var(
        model.factories, model.distribution_centers,
        domain=NonNegativeReals,
        doc="Final-product flow from factory j to distribution center k"
    )
    model.Z = Var(
        model.distribution_centers, model.customers,
        domain=NonNegativeReals,
        doc="Final-product flow from distribution center k to customer l"
    )
    model.Q = Var(model.factories, domain=Binary, doc="1 if factory j is opened")
    model.V = Var(model.distribution_centers, domain=Binary, doc="1 if DC k is opened")

    # Objective function: transportation costs + fixed opening costs
    def objective_rule(m):
        supplier_to_factory = sum(
            m.C1[i, j, t] * m.X[i, j, t]
            for i in m.suppliers
            for j in m.factories
            for t in m.components
        )
        factory_to_dc = sum(
            m.C2[j, k] * m.Y[j, k]
            for j in m.factories
            for k in m.distribution_centers
        )
        dc_to_customer = sum(
            m.C3[k, l] * m.Z[k, l]
            for k in m.distribution_centers
            for l in m.customers
        )
        factory_fixed_cost = sum(m.O[j] * m.Q[j] for j in m.factories)
        dc_fixed_cost = sum(m.S[k] * m.V[k] for k in m.distribution_centers)

        return (
            supplier_to_factory
            + factory_to_dc
            + dc_to_customer
            + factory_fixed_cost
            + dc_fixed_cost
        )

    model.total_cost = Objective(rule=objective_rule, sense=minimize)

    # Supplier component-capacity constraints
    def supplier_component_capacity_rule(m, i, t):
        return sum(m.X[i, j, t] for j in m.factories) <= m.a[i, t]

    model.supplier_component_capacity = Constraint(
        model.suppliers,
        model.components,
        rule=supplier_component_capacity_rule,
    )

    # Factory-capacity constraints activated by factory-opening binary variable
    def factory_capacity_rule(m, j):
        return sum(m.Y[j, k] for k in m.distribution_centers) <= m.b[j] * m.Q[j]

    model.factory_capacity = Constraint(model.factories, rule=factory_capacity_rule)

    # GAMS parameter P = 4
    model.max_open_factories = Constraint(
        expr=sum(model.Q[j] for j in model.factories) <= 4
    )

    # Distribution-center-capacity constraints activated by DC-opening binary variable
    def dc_capacity_rule(m, k):
        return sum(m.Z[k, l] for l in m.customers) <= m.c[k] * m.V[k]

    model.dc_capacity = Constraint(model.distribution_centers, rule=dc_capacity_rule)

    # GAMS parameter D = 4
    model.max_open_dcs = Constraint(
        expr=sum(model.V[k] for k in model.distribution_centers) <= 4
    )

    # Component balance at each factory:
    # incoming component t = W[t] * total final-product output of factory j
    def component_balance_rule(m, j, t):
        return (
            sum(m.X[i, j, t] for i in m.suppliers)
            - m.W[t] * sum(m.Y[j, k] for k in m.distribution_centers)
            == 0
        )

    model.component_balance = Constraint(
        model.factories,
        model.components,
        rule=component_balance_rule,
    )

    # Flow conservation at each distribution center
    def dc_flow_balance_rule(m, k):
        return (
            sum(m.Y[j, k] for j in m.factories)
            - sum(m.Z[k, l] for l in m.customers)
            == 0
        )

    model.dc_flow_balance = Constraint(
        model.distribution_centers,
        rule=dc_flow_balance_rule,
    )

    # Customer demand satisfaction
    def customer_demand_rule(m, l):
        return sum(m.Z[k, l] for k in m.distribution_centers) >= m.d[l]

    model.customer_demand = Constraint(model.customers, rule=customer_demand_rule)

    return model


def solve_model(solver_name="highs"):
    """Build and solve the MILP model."""
    model = build_model()
    solver = SolverFactory(solver_name)

    if not solver.available(False):
        raise RuntimeError(
            f"Solver '{solver_name}' is not available. "
            "Install HiGHS with `pip install highspy` or configure another MILP solver."
        )

    results = solver.solve(model, tee=False)
    return model, results


def print_solution(model):
    """Print nonzero decision variables and the optimal objective value."""
    print(f"Minimum total cost: {value(model.total_cost):.2f}\n")

    print("Open factories:")
    for j in model.factories:
        if value(model.Q[j]) > 0.5:
            print(f"  Factory {j}")

    print("\nOpen distribution centers:")
    for k in model.distribution_centers:
        if value(model.V[k]) > 0.5:
            print(f"  Distribution center {k}")

    print("\nSupplier -> Factory component flows:")
    for i in model.suppliers:
        for j in model.factories:
            for t in model.components:
                x = value(model.X[i, j, t])
                if x > 1e-7:
                    print(f"  Supplier {i} -> Factory {j}, Component {t}: {x:.2f}")

    print("\nFactory -> Distribution Center flows:")
    for j in model.factories:
        for k in model.distribution_centers:
            y = value(model.Y[j, k])
            if y > 1e-7:
                print(f"  Factory {j} -> DC {k}: {y:.2f}")

    print("\nDistribution Center -> Customer flows:")
    for k in model.distribution_centers:
        for l in model.customers:
            z = value(model.Z[k, l])
            if z > 1e-7:
                print(f"  DC {k} -> Customer {l}: {z:.2f}")


if __name__ == "__main__":
    solved_model, solve_results = solve_model("highs")
    print_solution(solved_model)
