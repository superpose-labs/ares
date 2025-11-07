"""Random sampling baseline for PSL problem.

This module implements a simple random sampling baseline that randomly
selects K items from the dataset without any optimization.
"""

import time
from typing import Any

import numpy as np

from ..psl import ProblemInstance


def random_sampling(
    instance: ProblemInstance, verbose: bool = True, seed: int = None
) -> dict[str, Any]:
    """
    Random sampling baseline that selects K items uniformly at random.

    This serves as a baseline to compare against optimization-based solvers.
    It does not consider coverage, diversity, or any other objective - it
    simply samples K items uniformly from the n available items.

    Args:
        instance: ProblemInstance with A, w, S, K, alpha
        verbose: If True, print status messages
        seed: Random seed for reproducibility. If None, uses current numpy random state.

    Returns:
        dict with:
            - selected_indices: np.ndarray of selected item indices
            - selected_rollout_ids: list of rollout IDs
            - runtime_seconds: float

    Complexity: O(K) for sampling + O(1) for everything else
    """
    start_time = time.time()

    n = instance.metadata["n"]  # Number of items
    K = instance.K  # Number of items to select

    if verbose:
        print(f"Running random sampling for K={K} items from n={n}")

    # Set random seed if provided
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Randomly sample K indices without replacement
    if K > n:
        if verbose:
            print(f"Warning: K={K} > n={n}, selecting all {n} items")
        selected_indices = np.arange(n)
    else:
        selected_indices = rng.choice(n, size=K, replace=False)

    # Sort for consistency
    selected_indices = np.sort(selected_indices)

    # Get rollout IDs
    selected_rollout_ids = [instance.rollout_ids[i] for i in selected_indices]

    end_time = time.time()
    runtime = end_time - start_time

    result = {
        "selected_indices": selected_indices,
        "selected_rollout_ids": selected_rollout_ids,
        "runtime_seconds": runtime,
    }

    if verbose:
        print(f"Random sampling completed in {runtime:.4f}s")
        print(f"Selected {len(selected_indices)} items")

    return result
