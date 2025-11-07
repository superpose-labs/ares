"""Greedy solver for PSL problem with diversity penalty.

This module implements a lazy greedy algorithm that maximizes coverage
while penalizing redundancy through a diversity constraint.
"""

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from ..psl import ProblemInstance


def lazy_greedy_with_diversity(
    instance: ProblemInstance, verbose: bool = True
) -> dict[str, Any]:
    """
    Greedy selection maximizing coverage while penalizing redundancy.

    The algorithm iteratively selects items that maximize marginal gain,
    where marginal gain = coverage_gain - alpha * diversity_penalty.

    Algorithm:
    1. Start with empty selection
    2. For each iteration:
       - For each candidate item k:
         * Compute coverage_gain: sum of weights for newly covered elements
         * Compute diversity_penalty: sum of similarities to already selected items
         * Compute marginal_gain = coverage_gain - alpha * diversity_penalty
       - Select item k with highest marginal gain
       - Update covered elements
    3. Continue until K items are selected

    Args:
        instance: ProblemInstance with A, w, S, K, alpha
        verbose: If True, show progress bar and print status

    Returns:
        dict with:
            - selected_indices: np.ndarray of selected item indices
            - selected_rollout_ids: list of rollout IDs
            - coverage_history: list of coverage scores per iteration
            - diversity_history: list of diversity penalties per iteration
            - marginal_gains_history: list of marginal gains per iteration
            - runtime_seconds: float

    Complexity: O(K * n * m) where n is number of items, m is number of elements
    """
    start_time = time.time()

    n = instance.metadata["n"]  # Number of items
    m = instance.metadata["m"]  # Number of elements

    # Initialize
    selected = []  # Indices of selected items
    covered = np.zeros(m, dtype=bool)  # Boolean array of covered elements
    coverage_history = []
    diversity_history = []
    marginal_gains_history = []

    # Convert sparse matrices to CSC format for efficient column access
    A_csc = instance.A.tocsc()
    S_csc = instance.S.tocsc()

    if verbose:
        print(f"Running greedy solver for K={instance.K} items from n={n}")
        pbar = tqdm(total=instance.K, desc="Greedy selection")

    for iteration in range(instance.K):
        best_k = None
        best_marginal_gain = -np.inf

        # Evaluate all candidate items
        for k in range(n):
            # Skip if already selected
            if k in selected:
                continue

            # Compute coverage gain
            # Get elements covered by item k
            covered_by_k = A_csc[:, k].toarray().flatten() > 0
            newly_covered = covered_by_k & ~covered
            coverage_gain = (instance.w * newly_covered).sum()

            # Compute diversity penalty (similarity to already selected items)
            diversity_penalty = 0.0
            if len(selected) > 0:
                # Get similarities between k and all selected items
                for j in selected:
                    sim = S_csc[k, j]
                    if hasattr(sim, "toarray"):  # If it's a sparse matrix element
                        sim = sim.toarray()[0, 0]
                    diversity_penalty += sim

            # Compute marginal gain
            marginal_gain = coverage_gain - instance.alpha * diversity_penalty

            # Update best candidate
            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_k = k
                best_coverage_gain = coverage_gain
                best_diversity_penalty = diversity_penalty

        # Select best item
        if best_k is None:
            if verbose:
                print(f"\nWarning: No valid item found at iteration {iteration}")
            break

        selected.append(best_k)

        # Update covered elements
        covered_by_best = A_csc[:, best_k].toarray().flatten() > 0
        covered = covered | covered_by_best

        # Track history
        current_coverage = (instance.w * covered).sum()
        coverage_history.append(float(current_coverage))
        diversity_history.append(float(best_diversity_penalty))
        marginal_gains_history.append(float(best_marginal_gain))

        if verbose:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "coverage": f"{current_coverage:.3f}",
                    "marginal": f"{best_marginal_gain:.3f}",
                }
            )

    if verbose:
        pbar.close()

    end_time = time.time()
    runtime = end_time - start_time

    # Convert to numpy array
    selected_indices = np.array(selected, dtype=int)

    # Get rollout IDs
    selected_rollout_ids = [instance.rollout_ids[i] for i in selected_indices]

    result = {
        "selected_indices": selected_indices,
        "selected_rollout_ids": selected_rollout_ids,
        "coverage_history": coverage_history,
        "diversity_history": diversity_history,
        "marginal_gains_history": marginal_gains_history,
        "runtime_seconds": runtime,
    }

    if verbose:
        final_coverage = coverage_history[-1] if coverage_history else 0.0
        print(f"Greedy solver completed in {runtime:.2f}s")
        print(f"Selected {len(selected_indices)} items")
        print(f"Final coverage: {final_coverage:.4f}")

    return result


def lazy_greedy_optimized(
    instance: ProblemInstance, verbose: bool = True
) -> dict[str, Any]:
    """
    Optimized lazy greedy with priority queue and lazy evaluation.

    This version uses lazy evaluation to avoid recomputing marginal gains
    for all items at each iteration. Instead, it maintains an upper bound
    on marginal gains and only recomputes when necessary.

    Note: For n=200, the simple version is already very fast (<1s), so this
    optimization may not be necessary. Included for completeness.

    Args:
        instance: ProblemInstance
        verbose: If True, show progress bar

    Returns:
        Same as lazy_greedy_with_diversity
    """
    import heapq

    start_time = time.time()

    n = instance.metadata["n"]
    m = instance.metadata["m"]

    # Initialize
    selected = []
    covered = np.zeros(m, dtype=bool)
    coverage_history = []
    diversity_history = []
    marginal_gains_history = []

    # Convert to CSC format
    A_csc = instance.A.tocsc()
    S_csc = instance.S.tocsc()

    # Initialize priority queue with upper bounds
    # Heap stores (-marginal_gain, iteration_computed, item_index)
    # Use negative marginal gain because heapq is a min-heap
    heap = []
    for k in range(n):
        # Upper bound: assume no diversity penalty
        covered_by_k = A_csc[:, k].toarray().flatten() > 0
        coverage_gain = (instance.w * covered_by_k).sum()
        heapq.heappush(heap, (-coverage_gain, 0, k))

    if verbose:
        print(f"Running optimized greedy solver for K={instance.K} items from n={n}")
        pbar = tqdm(total=instance.K, desc="Lazy greedy")

    current_iteration = 0

    while len(selected) < instance.K and heap:
        # Get item with highest upper bound
        neg_marginal, iter_computed, k = heapq.heappop(heap)

        # Skip if already selected
        if k in selected:
            continue

        # Recompute marginal gain if stale
        if iter_computed < current_iteration:
            # Compute coverage gain
            covered_by_k = A_csc[:, k].toarray().flatten() > 0
            newly_covered = covered_by_k & ~covered
            coverage_gain = (instance.w * newly_covered).sum()

            # Compute diversity penalty
            diversity_penalty = 0.0
            if len(selected) > 0:
                for j in selected:
                    sim = S_csc[k, j]
                    if hasattr(sim, "toarray"):
                        sim = sim.toarray()[0, 0]
                    diversity_penalty += sim

            # Compute marginal gain
            marginal_gain = coverage_gain - instance.alpha * diversity_penalty

            # Push back with updated values
            heapq.heappush(heap, (-marginal_gain, current_iteration, k))
            continue

        # Otherwise, this is the best item for this iteration
        marginal_gain = -neg_marginal

        # Compute actual values for tracking
        covered_by_k = A_csc[:, k].toarray().flatten() > 0
        newly_covered = covered_by_k & ~covered
        coverage_gain = (instance.w * newly_covered).sum()

        diversity_penalty = 0.0
        if len(selected) > 0:
            for j in selected:
                sim = S_csc[k, j]
                if hasattr(sim, "toarray"):
                    sim = sim.toarray()[0, 0]
                diversity_penalty += sim

        # Select this item
        selected.append(k)
        covered = covered | covered_by_k

        # Track history
        current_coverage = (instance.w * covered).sum()
        coverage_history.append(float(current_coverage))
        diversity_history.append(float(diversity_penalty))
        marginal_gains_history.append(float(marginal_gain))

        current_iteration += 1

        if verbose:
            pbar.update(1)
            pbar.set_postfix(
                {"coverage": f"{current_coverage:.3f}", "marginal": f"{marginal_gain:.3f}"}
            )

    if verbose:
        pbar.close()

    end_time = time.time()
    runtime = end_time - start_time

    selected_indices = np.array(selected, dtype=int)
    selected_rollout_ids = [instance.rollout_ids[i] for i in selected_indices]

    result = {
        "selected_indices": selected_indices,
        "selected_rollout_ids": selected_rollout_ids,
        "coverage_history": coverage_history,
        "diversity_history": diversity_history,
        "marginal_gains_history": marginal_gains_history,
        "runtime_seconds": runtime,
    }

    if verbose:
        final_coverage = coverage_history[-1] if coverage_history else 0.0
        print(f"Optimized greedy solver completed in {runtime:.2f}s")
        print(f"Selected {len(selected_indices)} items")
        print(f"Final coverage: {final_coverage:.4f}")

    return result
