"""Construct coverage taxonomy via window-based clustering.

This module provides functions to cluster sub-trajectory window embeddings
and construct a coverage matrix A and element weights w using IDF-style weighting.

Window-based clustering captures fine-grained behaviors within trajectories,
enabling many-to-many mapping where each rollout can cover multiple behavior
clusters and each cluster can be covered by multiple rollouts.
"""

from typing import Optional, List

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans


def build_window_coverage_taxonomy(
    window_embeddings: np.ndarray,
    window_to_rollout: List[str],
    rollout_ids: List[str],
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> dict:
    """Build coverage taxonomy by clustering sub-trajectory windows.

    This function clusters windows instead of whole trajectories, enabling
    many-to-many mapping: each rollout can cover multiple behavior clusters,
    and each cluster can be covered by multiple rollouts.

    Key difference from build_coverage_taxonomy():
    - Input: N_w windows (many per rollout) instead of n rollouts
    - Output: A[i,k] = 1 if rollout k has ANY window in cluster i
    - Result: Non-trivial coverage problem (not all clusters covered by default)

    Args:
        window_embeddings: np.ndarray of shape (N_w, d) - all window embeddings
        window_to_rollout: List[str] of length N_w - rollout ID for each window
        rollout_ids: List[str] of length n - all unique rollout IDs
        n_clusters: int or None - number of clusters. If None, use k = sqrt(N_w)
        random_state: int - random seed for k-means

    Returns:
        Dictionary with:
            - A: scipy.sparse.csr_matrix of shape (m, n) - coverage matrix
            - w: np.ndarray of shape (m,) - element weights (sum = 1.0)
            - cluster_labels: np.ndarray of shape (N_w,) - cluster per window
            - cluster_sizes: np.ndarray of shape (m,) - windows per cluster
            - n_clusters: int - number of clusters
            - cluster_centers: np.ndarray of shape (m, d) - cluster centroids
            - window_to_rollout: List[str] - original mapping (pass through)
            - rollout_coverage_counts: np.ndarray of shape (n,) - clusters per rollout

    Example:
        200 rollouts × 19 windows = 3,800 total windows
        Cluster into m=20 behavior clusters
        → A is (20, 200) with ~800 non-zeros (avg 4 clusters per rollout)
    """
    N_w = len(window_embeddings)
    n = len(rollout_ids)

    if n_clusters is None:
        n_clusters = int(np.sqrt(N_w))
        print(f"Auto-selecting n_clusters = sqrt({N_w}) = {n_clusters}")

    # Validate inputs
    assert len(window_to_rollout) == N_w, (
        f"window_to_rollout length {len(window_to_rollout)} != {N_w}"
    )
    assert n_clusters > 0, f"n_clusters must be positive, got {n_clusters}"
    assert n_clusters <= N_w, f"n_clusters {n_clusters} > N_w {N_w}"

    # Run k-means clustering on windows
    print(f"Running k-means on {N_w} windows with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(window_embeddings)

    # Count cluster sizes (number of windows per cluster)
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)

    # Build coverage matrix A: A[i, k] = 1 if rollout k has any window in cluster i
    # This is the key difference: many-to-many mapping
    print(f"Building coverage matrix A ({n_clusters}, {n})...")

    # Create rollout ID to index mapping
    rollout_to_idx = {rid: idx for idx, rid in enumerate(rollout_ids)}

    # Track which (cluster, rollout) pairs have coverage
    coverage_pairs = set()  # Set of (cluster_idx, rollout_idx) tuples

    for window_idx, (cluster_idx, rollout_id) in enumerate(
        zip(cluster_labels, window_to_rollout)
    ):
        rollout_idx = rollout_to_idx[rollout_id]
        coverage_pairs.add((cluster_idx, rollout_idx))

    # Convert to sparse matrix format
    row_indices = []
    col_indices = []
    for cluster_idx, rollout_idx in coverage_pairs:
        row_indices.append(cluster_idx)
        col_indices.append(rollout_idx)

    data = np.ones(len(row_indices))  # Binary coverage
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_clusters, n))

    # Compute IDF-style weights based on window counts
    # w_i = log(1 + N_w / cluster_size_i)
    w = np.log(1 + N_w / cluster_sizes)

    # Normalize weights to sum to 1.0
    w = w / w.sum()

    # Compute coverage statistics
    rollout_coverage_counts = np.array(A.sum(axis=0)).flatten()  # Clusters per rollout

    print(f"Built coverage matrix A: {A.shape}, nnz={A.nnz}")
    print(f"Coverage density: {A.nnz / (n_clusters * n) * 100:.2f}%")
    print(f"Avg clusters per rollout: {rollout_coverage_counts.mean():.2f}")
    print(f"Window cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}, std={cluster_sizes.std():.1f}")
    print(f"Weight range: min={w.min():.4f}, max={w.max():.4f}, sum={w.sum():.4f}")

    return {
        "A": A,
        "w": w,
        "cluster_labels": cluster_labels,
        "cluster_sizes": cluster_sizes,
        "n_clusters": n_clusters,
        "cluster_centers": kmeans.cluster_centers_,
        "window_to_rollout": window_to_rollout,
        "rollout_coverage_counts": rollout_coverage_counts,
    }


def get_window_cluster_statistics(
    cluster_labels: np.ndarray,
    cluster_sizes: np.ndarray,
    window_to_rollout: List[str],
    rollout_ids: List[str],
    A: sp.csr_matrix,
    top_k: int = 5,
) -> dict:
    """Get detailed statistics about window-based clusters.

    Args:
        cluster_labels: np.ndarray of shape (N_w,) - cluster assignments for windows
        cluster_sizes: np.ndarray of shape (m,) - windows per cluster
        window_to_rollout: List[str] of length N_w - rollout ID for each window
        rollout_ids: List[str] of length n - all rollout IDs
        A: Coverage matrix (m, n)
        top_k: int - number of top clusters to report

    Returns:
        Dictionary with cluster statistics
    """
    n_clusters = len(cluster_sizes)
    N_w = len(cluster_labels)
    n = len(rollout_ids)

    # Rollout coverage counts
    rollout_coverage_counts = np.array(A.sum(axis=0)).flatten()

    # Sort clusters by size
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    top_cluster_sizes = cluster_sizes[sorted_indices[:top_k]]

    stats = {
        "n_clusters": n_clusters,
        "n_windows": N_w,
        "n_rollouts": n,
        "avg_windows_per_rollout": N_w / n,
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "cluster_size_mean": float(cluster_sizes.mean()),
        "cluster_size_median": float(np.median(cluster_sizes)),
        "cluster_size_std": float(cluster_sizes.std()),
        "top_cluster_sizes": top_cluster_sizes.tolist(),
        "top_cluster_ids": sorted_indices[:top_k].tolist(),
        "coverage_nnz": A.nnz,
        "coverage_density": A.nnz / (n_clusters * n),
        "avg_clusters_per_rollout": rollout_coverage_counts.mean(),
        "min_clusters_per_rollout": int(rollout_coverage_counts.min()),
        "max_clusters_per_rollout": int(rollout_coverage_counts.max()),
    }

    # Get example rollouts from different clusters
    rollout_to_idx = {rid: idx for idx, rid in enumerate(rollout_ids)}
    examples = {}

    for i in range(min(3, n_clusters)):
        cluster_idx = sorted_indices[i]

        # Find rollouts that have windows in this cluster
        rollout_indices = A[cluster_idx, :].nonzero()[1]
        cluster_rollouts = [rollout_ids[idx] for idx in rollout_indices]

        # Count windows per rollout in this cluster
        window_counts = {}
        for window_idx, (label, rollout_id) in enumerate(
            zip(cluster_labels, window_to_rollout)
        ):
            if label == cluster_idx:
                window_counts[rollout_id] = window_counts.get(rollout_id, 0) + 1

        # Sort by window count
        top_rollouts = sorted(
            window_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        examples[f"cluster_{cluster_idx}_size_{cluster_sizes[cluster_idx]}"] = [
            f"{rid} ({count} windows)" for rid, count in top_rollouts
        ]

    stats["example_rollouts"] = examples

    return stats


def print_window_cluster_statistics(stats: dict) -> None:
    """Print formatted window cluster statistics.

    Args:
        stats: Dictionary from get_window_cluster_statistics()
    """
    print("\n" + "=" * 60)
    print("Window-Based Cluster Statistics")
    print("=" * 60)

    print(f"\nOverall:")
    print(f"  Number of clusters:         {stats['n_clusters']}")
    print(f"  Number of windows:          {stats['n_windows']}")
    print(f"  Number of rollouts:         {stats['n_rollouts']}")
    print(f"  Avg windows per rollout:    {stats['avg_windows_per_rollout']:.1f}")

    print(f"\nCluster Sizes (windows):")
    print(f"  Min:     {stats['cluster_size_min']}")
    print(f"  Max:     {stats['cluster_size_max']}")
    print(f"  Mean:    {stats['cluster_size_mean']:.1f}")
    print(f"  Median:  {stats['cluster_size_median']:.1f}")
    print(f"  Std:     {stats['cluster_size_std']:.1f}")

    print(f"\nCoverage Matrix:")
    print(f"  Shape:   ({stats['n_clusters']}, {stats['n_rollouts']})")
    print(f"  Non-zeros:  {stats['coverage_nnz']}")
    print(f"  Density:    {stats['coverage_density'] * 100:.2f}%")

    print(f"\nClusters per Rollout:")
    print(f"  Min:     {stats['min_clusters_per_rollout']}")
    print(f"  Max:     {stats['max_clusters_per_rollout']}")
    print(f"  Mean:    {stats['avg_clusters_per_rollout']:.2f}")

    print(f"\nTop {len(stats['top_cluster_sizes'])} Largest Clusters:")
    for i, (cluster_id, size) in enumerate(
        zip(stats['top_cluster_ids'], stats['top_cluster_sizes'])
    ):
        print(f"  {i+1}. Cluster {cluster_id}: {size} windows")

    print(f"\nExample Rollouts from Different Clusters:")
    for cluster_name, rollout_info in stats["example_rollouts"].items():
        print(f"  {cluster_name}:")
        for info in rollout_info:
            print(f"    - {info}")

    print("=" * 60 + "\n")
