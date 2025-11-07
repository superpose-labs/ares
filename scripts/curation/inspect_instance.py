#!/usr/bin/env python3
"""Inspect a QUBO problem instance.

This script loads a saved ProblemInstance and provides detailed inspection
and visualization of its components.

Usage:
    python scripts/curation/inspect_instance.py [INSTANCE_PATH]

Example:
    python scripts/curation/inspect_instance.py data/curation/kaist_instance.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from superpose_data_engine.psl import ProblemInstance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect a QUBO problem instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "instance_path",
        type=str,
        nargs="?",
        default="data/curation/kaist_instance.pkl",
        help="Path to problem instance pickle file",
    )

    parser.add_argument(
        "--show-rollouts",
        action="store_true",
        help="Show rollout IDs from each cluster",
    )

    parser.add_argument(
        "--show-similarities",
        type=int,
        default=0,
        metavar="N",
        help="Show top-N most similar pairs",
    )

    return parser.parse_args()


def show_cluster_rollouts(instance: ProblemInstance, max_clusters: int = 5):
    """Show rollout IDs organized by cluster.

    Args:
        instance: ProblemInstance to inspect
        max_clusters: Maximum number of clusters to display
    """
    print("\n" + "=" * 70)
    print("ROLLOUTS BY CLUSTER")
    print("=" * 70)

    n = instance.metadata['n']
    m = instance.metadata['m']

    # Extract cluster assignments from A
    # A[i, k] = 1 if rollout k is in cluster i
    A = instance.A.tocsr()

    # Get cluster sizes
    cluster_sizes = np.array(A.sum(axis=1)).flatten()
    sorted_cluster_indices = np.argsort(cluster_sizes)[::-1]

    for rank, cluster_idx in enumerate(sorted_cluster_indices[:max_clusters]):
        cluster_size = int(cluster_sizes[cluster_idx])
        weight = instance.w[cluster_idx]

        # Get rollouts in this cluster
        rollout_indices = A[cluster_idx].nonzero()[1]
        rollout_ids = [instance.rollout_ids[i] for i in rollout_indices]

        print(f"\nCluster {cluster_idx} (rank {rank+1}/{m}):")
        print(f"  Size:   {cluster_size} items")
        print(f"  Weight: {weight:.4f}")
        print(f"  Rollouts:")
        for i, rid in enumerate(rollout_ids[:10]):  # Show first 10
            print(f"    {i+1}. {rid}")
        if len(rollout_ids) > 10:
            print(f"    ... and {len(rollout_ids) - 10} more")

    print("=" * 70)


def show_top_similarities(instance: ProblemInstance, top_n: int = 10):
    """Show the most similar pairs of rollouts.

    Args:
        instance: ProblemInstance to inspect
        top_n: Number of top pairs to show
    """
    print("\n" + "=" * 70)
    print(f"TOP {top_n} MOST SIMILAR PAIRS")
    print("=" * 70)

    S = instance.S.tocoo()  # Convert to COO format for easy iteration

    # Get upper triangle only (avoid duplicates)
    mask = S.row < S.col
    rows = S.row[mask]
    cols = S.col[mask]
    sims = S.data[mask]

    # Sort by similarity
    sorted_indices = np.argsort(sims)[::-1][:top_n]

    for rank, idx in enumerate(sorted_indices):
        i, j = rows[idx], cols[idx]
        sim = sims[idx]
        rid_i = instance.rollout_ids[i]
        rid_j = instance.rollout_ids[j]

        print(f"\n{rank+1}. Similarity: {sim:.4f}")
        print(f"   Rollout {i}: {rid_i}")
        print(f"   Rollout {j}: {rid_j}")

    print("=" * 70)


def show_coverage_analysis(instance: ProblemInstance):
    """Analyze coverage properties.

    Args:
        instance: ProblemInstance to inspect
    """
    print("\n" + "=" * 70)
    print("COVERAGE ANALYSIS")
    print("=" * 70)

    n = instance.metadata['n']
    m = instance.metadata['m']
    A = instance.A.tocsr()

    # Items per cluster
    cluster_sizes = np.array(A.sum(axis=1)).flatten()

    # Clusters per item (should be 1 for our simplified version)
    item_clusters = np.array(A.sum(axis=0)).flatten()

    print(f"\nCluster Statistics:")
    print(f"  Total clusters:        {m}")
    print(f"  Cluster size range:    [{cluster_sizes.min()}, {cluster_sizes.max()}]")
    print(f"  Mean cluster size:     {cluster_sizes.mean():.1f}")
    print(f"  Median cluster size:   {np.median(cluster_sizes):.1f}")
    print(f"  Std cluster size:      {cluster_sizes.std():.1f}")

    print(f"\nItem Coverage:")
    print(f"  Total items:           {n}")
    print(f"  Items in 1 cluster:    {np.sum(item_clusters == 1)}")
    print(f"  Items in 0 clusters:   {np.sum(item_clusters == 0)}")
    print(f"  Items in >1 clusters:  {np.sum(item_clusters > 1)}")

    # Weight distribution
    print(f"\nWeight Distribution:")
    print(f"  Total weight:          {instance.w.sum():.6f}")
    print(f"  Weight range:          [{instance.w.min():.4f}, {instance.w.max():.4f}]")
    print(f"  Weight mean:           {instance.w.mean():.4f}")
    print(f"  Weight std:            {instance.w.std():.4f}")

    # Top weighted clusters
    top_indices = np.argsort(instance.w)[::-1][:5]
    print(f"\nTop 5 Weighted Clusters:")
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1}. Cluster {idx}: weight={instance.w[idx]:.4f}, size={int(cluster_sizes[idx])}")

    print("=" * 70)


def main():
    """Main inspection function."""
    args = parse_args()

    # Load instance
    print(f"Loading instance from: {args.instance_path}")
    instance = ProblemInstance.load(args.instance_path)

    # Print basic statistics
    instance.print_statistics()

    # Coverage analysis
    show_coverage_analysis(instance)

    # Show rollouts by cluster
    if args.show_rollouts:
        show_cluster_rollouts(instance, max_clusters=5)

    # Show top similarities
    if args.show_similarities > 0:
        show_top_similarities(instance, top_n=args.show_similarities)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
