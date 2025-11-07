"""CP-SAT solver for PSL problem using Google OR-Tools.

This module implements an exact solver using constraint programming with
quadratic penalty linearization for the diversity term.
"""

import time
from typing import Any

import numpy as np
from ortools.sat.python import cp_model

from ..psl import ProblemInstance


def solve_cpsat(
    instance: ProblemInstance,
    time_limit_seconds: int = 300,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Solve PSL using Google OR-Tools CP-SAT solver with quadratic diversity penalty.

    The problem is formulated as:
        maximize    sum(w[i] * z[i] for i in elements)
                    - alpha * sum(S[k1,k2] * y[k1] * y[k2] for k1,k2 pairs)
        subject to:
            z[i] <= sum(A[i,k] * y[k] for k in items)  # Coverage linking
            sum(y) == K  # Cardinality constraint
            y[k], z[i] in {0, 1}

    The quadratic diversity penalty is linearized using auxiliary variables:
    For each pair (k1, k2) with non-zero similarity:
        - Create p[k1,k2] = y[k1] * y[k2]
        - Enforce with constraints: p <= y[k1], p <= y[k2], p >= y[k1] + y[k2] - 1

    Args:
        instance: ProblemInstance with all problem data
        time_limit_seconds: Maximum time for solver (default: 300s)
        verbose: If True, print solver progress

    Returns:
        dict with:
            - selected_indices: np.ndarray of selected item indices
            - selected_rollout_ids: list of rollout IDs
            - objective_value: float (coverage score - alpha * diversity penalty)
            - solve_status: str ('optimal', 'feasible', 'infeasible', 'timeout')
            - runtime_seconds: float
            - num_auxiliary_vars: int (number of pairwise product variables)
            - gap: float (optimality gap if not optimal)

    Note: CP-SAT requires integer coefficients, so weights are scaled by 10000
    """
    start_time = time.time()

    n = instance.metadata["n"]
    m = instance.metadata["m"]

    if verbose:
        print(f"Setting up CP-SAT model for n={n}, m={m}, K={instance.K}")

    model = cp_model.CpModel()

    # Decision variables
    y = [model.NewBoolVar(f"y_{k}") for k in range(n)]
    z = [model.NewBoolVar(f"z_{i}") for i in range(m)]

    # Cardinality constraint
    model.Add(sum(y) == instance.K)

    # Coverage constraints
    A_csc = instance.A.tocsc()
    for i in range(m):
        items_covering_i = A_csc[i, :].nonzero()[1]
        if len(items_covering_i) > 0:
            model.Add(z[i] <= sum(y[k] for k in items_covering_i))
        else:
            model.Add(z[i] == 0)

    # Diversity penalty: sum(S[k1,k2] * y[k1] * y[k2])
    # Create auxiliary variables p[k1,k2] = y[k1] * y[k2]
    if verbose:
        print("Adding diversity penalty (linearized quadratic term)...")

    scale_factor = 10000
    S_coo = instance.S.tocoo()

    penalty_terms = []
    num_pairs = 0

    for idx in range(len(S_coo.data)):
        k1, k2 = S_coo.row[idx], S_coo.col[idx]
        sim = S_coo.data[idx]

        # Only upper triangle
        if k1 < k2 and sim > 0:
            # Create auxiliary variable for product
            p = model.NewBoolVar(f"p_{k1}_{k2}")

            # Linearization: p = y[k1] * y[k2]
            # p <= y[k1]
            # p <= y[k2]
            # p >= y[k1] + y[k2] - 1
            model.Add(p <= y[k1])
            model.Add(p <= y[k2])
            model.Add(p >= y[k1] + y[k2] - 1)

            # Add to penalty
            scaled_sim = int(sim * instance.alpha * scale_factor)
            penalty_terms.append(scaled_sim * p)
            num_pairs += 1

    if verbose:
        print(f"Added {num_pairs} auxiliary variables for diversity penalty")

    # Objective: maximize coverage - alpha * diversity_penalty
    scaled_weights = (instance.w * scale_factor).astype(int)
    coverage_terms = [scaled_weights[i] * z[i] for i in range(m)]

    if penalty_terms:
        model.Maximize(sum(coverage_terms) - sum(penalty_terms))
    else:
        model.Maximize(sum(coverage_terms))

    if verbose:
        print(f"Starting solver with time limit {time_limit_seconds}s...")

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = verbose

    solve_start = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - solve_start

    # Extract solution (same as before)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        selected_indices = np.array([k for k in range(n) if solver.Value(y[k]) == 1])
        covered = np.array([solver.Value(z[i]) for i in range(m)])

        # Compute coverage
        coverage_score = float((instance.w * covered).sum())

        # Compute diversity penalty
        diversity_penalty = 0.0
        for idx in range(len(S_coo.data)):
            k1, k2 = S_coo.row[idx], S_coo.col[idx]
            if k1 < k2 and solver.Value(y[k1]) and solver.Value(y[k2]):
                diversity_penalty += S_coo.data[idx]

        objective_value = coverage_score - instance.alpha * diversity_penalty

        selected_rollout_ids = [instance.rollout_ids[k] for k in selected_indices]
        solve_status = "optimal" if status == cp_model.OPTIMAL else "feasible"

        gap = 0.0
        if status == cp_model.FEASIBLE and solver.BestObjectiveBound():
            best_bound = solver.BestObjectiveBound() / scale_factor
            gap = abs(best_bound - objective_value) / objective_value if objective_value > 0 else 0.0

    else:
        solve_status = "infeasible" if status == cp_model.INFEASIBLE else "timeout"
        selected_indices = np.array([], dtype=int)
        selected_rollout_ids = []
        objective_value = 0.0
        gap = 1.0

    total_runtime = time.time() - start_time

    result = {
        "selected_indices": selected_indices,
        "selected_rollout_ids": selected_rollout_ids,
        "objective_value": objective_value,
        "solve_status": solve_status,
        "runtime_seconds": total_runtime,
        "solve_time_seconds": solve_time,
        "num_auxiliary_vars": num_pairs,
        "gap": gap,
    }

    if verbose:
        print(f"\nSolver finished with status: {solve_status}")
        if len(selected_indices) > 0:
            print(f"Selected {len(selected_indices)} items")
            print(f"Objective value: {objective_value:.4f}")

    return result
