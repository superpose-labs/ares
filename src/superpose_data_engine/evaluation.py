"""Evaluation metrics for PSL solutions.

This module provides functions to evaluate the quality of data curation
solutions by measuring coverage, redundancy, and diversity metrics.
"""

from typing import Any

import numpy as np
import scipy.sparse as sp

from .psl import ProblemInstance


def evaluate_solution(
    instance: ProblemInstance, selected_indices: np.ndarray, verbose: bool = False
) -> dict[str, Any]:
    """
    Compute quality metrics for a data curation solution.

    Metrics computed:
    - coverage_score: sum(w[i] * z[i]) where z[i] = 1 if element i is covered
    - coverage_fraction: fraction of elements covered (# covered / m)
    - num_covered_elements: number of elements covered
    - avg_redundancy: mean pairwise similarity of selected items
    - max_redundancy: maximum pairwise similarity of selected items
    - min_redundancy: minimum pairwise similarity of selected items
    - total_similarity: sum of all pairwise similarities
    - cardinality: number of items selected
    - objective_value: coverage_score - alpha * total_similarity
    - cluster_coverage: distribution of coverage per element (for analysis)
    - coverage_uniformity: how uniformly clusters are covered

    Args:
        instance: ProblemInstance with all problem data
        selected_indices: np.ndarray of selected item indices (length K)
        verbose: If True, print detailed metrics

    Returns:
        dict with all computed metrics
    """
    n = instance.metadata["n"]
    m = instance.metadata["m"]

    # Handle empty selection
    if len(selected_indices) == 0:
        return {
            "coverage_score": 0.0,
            "coverage_fraction": 0.0,
            "num_covered_elements": 0,
            "avg_redundancy": 0.0,
            "max_redundancy": 0.0,
            "min_redundancy": 0.0,
            "total_similarity": 0.0,
            "cardinality": 0,
            "objective_value": 0.0,
            "cluster_coverage": np.zeros(m),
            "coverage_uniformity": 0.0,
        }

    # Create selection vector
    y = np.zeros(n, dtype=int)
    y[selected_indices] = 1

    # Compute coverage
    # z[i] = 1 if element i is covered by at least one selected item
    covered = (instance.A @ y) > 0  # Boolean array of shape (m,)

    # Coverage score: weighted sum of covered elements
    coverage_score = float((instance.w * covered).sum())

    # Coverage fraction
    num_covered_elements = int(covered.sum())
    coverage_fraction = num_covered_elements / m if m > 0 else 0.0

    # Cluster coverage distribution (how many items cover each element)
    cluster_coverage_counts = np.asarray(instance.A @ y).flatten()

    # Coverage uniformity: measure how uniformly elements are covered
    # Use coefficient of variation (std/mean) - lower is more uniform
    if num_covered_elements > 0:
        covered_counts = cluster_coverage_counts[covered]
        coverage_uniformity = float(covered_counts.std() / covered_counts.mean())
    else:
        coverage_uniformity = 0.0

    # Compute redundancy metrics
    similarities = []

    # Iterate over all pairs of selected items
    for i, k1 in enumerate(selected_indices):
        for k2 in selected_indices[i + 1 :]:
            # Get similarity from sparse matrix
            sim = instance.S[k1, k2]

            # Handle sparse matrix element
            if sp.issparse(instance.S):
                if hasattr(sim, "toarray"):
                    sim = sim.toarray()[0, 0]
                elif hasattr(sim, "todense"):
                    sim = sim.todense()[0, 0]

            # Only include non-zero similarities
            if sim > 1e-10:
                similarities.append(float(sim))

    # Compute redundancy statistics
    if similarities:
        avg_redundancy = float(np.mean(similarities))
        max_redundancy = float(np.max(similarities))
        min_redundancy = float(np.min(similarities))
        total_similarity = float(np.sum(similarities))
    else:
        avg_redundancy = 0.0
        max_redundancy = 0.0
        min_redundancy = 0.0
        total_similarity = 0.0

    # Cardinality
    cardinality = int(len(selected_indices))

    # Objective value: coverage - alpha * diversity_penalty
    objective_value = coverage_score - instance.alpha * total_similarity

    # Assemble results
    metrics = {
        "coverage_score": coverage_score,
        "coverage_fraction": coverage_fraction,
        "num_covered_elements": num_covered_elements,
        "avg_redundancy": avg_redundancy,
        "max_redundancy": max_redundancy,
        "min_redundancy": min_redundancy,
        "total_similarity": total_similarity,
        "cardinality": cardinality,
        "objective_value": objective_value,
        "cluster_coverage_counts": cluster_coverage_counts.tolist(),
        "coverage_uniformity": coverage_uniformity,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Solution Evaluation Metrics")
        print("=" * 60)
        print(f"Cardinality:              {cardinality} / {instance.K} (target)")
        print(f"\nCoverage Metrics:")
        print(f"  Coverage score:         {coverage_score:.4f}")
        print(f"  Elements covered:       {num_covered_elements} / {m} ({coverage_fraction:.1%})")
        print(f"  Coverage uniformity:    {coverage_uniformity:.4f}")
        print(f"\nDiversity Metrics:")
        print(f"  Avg pairwise similarity: {avg_redundancy:.4f}")
        print(f"  Max pairwise similarity: {max_redundancy:.4f}")
        print(f"  Min pairwise similarity: {min_redundancy:.4f}")
        print(f"  Total similarity:        {total_similarity:.4f}")
        print(f"  Num pairs evaluated:     {len(similarities)}")
        print(f"\nObjective:")
        print(f"  Objective value:         {objective_value:.4f}")
        print(f"  Coverage term:           {coverage_score:.4f}")
        print(f"  Diversity penalty:       {instance.alpha * total_similarity:.4f}")
        print("=" * 60 + "\n")

    return metrics


def compare_solutions(
    instance: ProblemInstance,
    solutions: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compare multiple solutions side-by-side.

    Args:
        instance: ProblemInstance
        solutions: dict mapping solver name to solution dict
                   Each solution dict should have 'selected_indices' key
        verbose: If True, print comparison table

    Returns:
        dict with:
            - metrics: dict mapping solver name to evaluation metrics
            - best_by_objective: solver name with best objective value
            - best_by_coverage: solver name with best coverage score
    """
    import pandas as pd

    metrics = {}

    # Evaluate each solution
    for solver_name, solution in solutions.items():
        selected_indices = solution.get("selected_indices", np.array([]))
        metrics[solver_name] = evaluate_solution(
            instance, selected_indices, verbose=False
        )

    # Create comparison dataframe
    if verbose:
        comparison_data = {}
        for solver_name, solver_metrics in metrics.items():
            comparison_data[solver_name] = {
                "Coverage Score": f"{solver_metrics['coverage_score']:.4f}",
                "Coverage %": f"{solver_metrics['coverage_fraction']:.1%}",
                "Avg Redundancy": f"{solver_metrics['avg_redundancy']:.4f}",
                "Max Redundancy": f"{solver_metrics['max_redundancy']:.4f}",
                "Objective": f"{solver_metrics['objective_value']:.4f}",
                "Cardinality": solver_metrics["cardinality"],
                "Runtime (s)": f"{solution.get('runtime_seconds', 0):.2f}",
            }

        df = pd.DataFrame(comparison_data).T
        print("\n" + "=" * 80)
        print("Solution Comparison")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80 + "\n")

    # Identify best solutions
    best_by_objective = max(
        metrics.keys(), key=lambda s: metrics[s]["objective_value"]
    )
    best_by_coverage = max(metrics.keys(), key=lambda s: metrics[s]["coverage_score"])

    if verbose:
        print(f"Best by objective value: {best_by_objective}")
        print(f"Best by coverage score:  {best_by_coverage}\n")

    return {
        "metrics": metrics,
        "best_by_objective": best_by_objective,
        "best_by_coverage": best_by_coverage,
    }


def compute_selection_overlap(
    solution1: dict[str, Any],
    solution2: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Compute overlap statistics between two solutions.

    Args:
        solution1: First solution dict with 'selected_indices'
        solution2: Second solution dict with 'selected_indices'
        verbose: If True, print overlap stats

    Returns:
        dict with:
            - jaccard_similarity: |intersection| / |union|
            - overlap_count: |intersection|
            - unique_to_1: number of items only in solution 1
            - unique_to_2: number of items only in solution 2
    """
    indices1 = set(solution1["selected_indices"])
    indices2 = set(solution2["selected_indices"])

    intersection = indices1 & indices2
    union = indices1 | indices2

    overlap_count = len(intersection)
    jaccard = overlap_count / len(union) if len(union) > 0 else 0.0
    unique_to_1 = len(indices1 - indices2)
    unique_to_2 = len(indices2 - indices1)

    result = {
        "jaccard_similarity": jaccard,
        "overlap_count": overlap_count,
        "unique_to_1": unique_to_1,
        "unique_to_2": unique_to_2,
    }

    if verbose:
        print(f"Selection Overlap:")
        print(f"  Jaccard similarity: {jaccard:.4f}")
        print(f"  Overlap count:      {overlap_count}")
        print(f"  Unique to sol 1:    {unique_to_1}")
        print(f"  Unique to sol 2:    {unique_to_2}")

    return result


def evaluate_random_baseline(
    instance: ProblemInstance, num_trials: int = 10, seed: int = 42
) -> dict[str, Any]:
    """
    Evaluate random selection baseline to compare against solvers.

    Args:
        instance: ProblemInstance
        num_trials: Number of random trials to average over
        seed: Random seed for reproducibility

    Returns:
        dict with:
            - mean_coverage_score: average coverage across trials
            - std_coverage_score: standard deviation
            - mean_objective: average objective value
            - mean_redundancy: average pairwise similarity
            - trials: list of metrics for each trial
    """
    np.random.seed(seed)

    n = instance.metadata["n"]
    trials_metrics = []

    for trial in range(num_trials):
        # Random selection
        selected_indices = np.random.choice(n, size=instance.K, replace=False)

        # Evaluate
        metrics = evaluate_solution(instance, selected_indices, verbose=False)
        trials_metrics.append(metrics)

    # Aggregate statistics
    coverage_scores = [m["coverage_score"] for m in trials_metrics]
    objectives = [m["objective_value"] for m in trials_metrics]
    redundancies = [m["avg_redundancy"] for m in trials_metrics]

    result = {
        "mean_coverage_score": float(np.mean(coverage_scores)),
        "std_coverage_score": float(np.std(coverage_scores)),
        "mean_objective": float(np.mean(objectives)),
        "std_objective": float(np.std(objectives)),
        "mean_redundancy": float(np.mean(redundancies)),
        "std_redundancy": float(np.std(redundancies)),
        "trials": trials_metrics,
    }

    return result
