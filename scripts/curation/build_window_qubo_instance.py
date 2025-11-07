#!/usr/bin/env python3
"""Build QUBO problem instance using window-based clustering.

This script clusters sub-trajectory windows to identify fine-grained behaviors,
creating a many-to-many coverage mapping between rollouts and behavior clusters.

Approach:
- Extract sliding windows from each trajectory (W=10 timesteps, S=5 stride)
- Cluster N_w=3,800 windows into m=20 behavior clusters
- Build coverage matrix A where A[i,k] = 1 if rollout k has any window in cluster i
- Result: Each rollout covers multiple behaviors (avg 6-8 clusters)

Usage:
    python scripts/curation/build_window_qubo_instance.py [OPTIONS]

Example:
    python scripts/curation/build_window_qubo_instance.py \
        --window-size 10 --stride 5 --n-clusters 20 --K 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from superpose_data_engine.psl import ProblemInstance
from superpose_data_engine.embeddings import load_kaist_trajectories, load_kaist_embeddings
from superpose_data_engine.windowing import TrajectoryWindower, WindowConfig, compute_window_statistics
from superpose_data_engine.taxonomy import (
    build_window_coverage_taxonomy,
    get_window_cluster_statistics,
    print_window_cluster_statistics,
)
from superpose_data_engine.similarity import (
    build_similarity_matrix,
    get_similarity_statistics,
    print_similarity_statistics,
    validate_similarity_matrix,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build QUBO problem instance using window-based clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="KAIST Nonprehensile Objects",
        help="Dataset formal name",
    )

    parser.add_argument(
        "--robot",
        type=str,
        default="Franka",
        help="Robot embodiment",
    )

    # Windowing parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Window size (number of timesteps per window)",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride between windows (50%% overlap with stride=window_size/2)",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="state",
        choices=["state", "action", "both"],
        help="Which trajectory data to use for windowing",
    )

    # Clustering parameters
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=50,
        help="Number of behavior clusters (default: 50)",
    )

    # Similarity matrix parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Top-k neighbors for similarity matrix (uses pooled embeddings)",
    )

    # Problem parameters
    parser.add_argument(
        "--K",
        type=int,
        default=60,
        help="Target cardinality (number of items to select)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Diversity penalty weight",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Budget penalty weight",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/curation/kaist_window_instance.pkl",
        help="Output path for problem instance",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clustering",
    )

    return parser.parse_args()


def main():
    """Main function to build window-based QUBO instance."""
    args = parse_args()

    print("=" * 70)
    print("BUILDING WINDOW-BASED QUBO PROBLEM INSTANCE")
    print("=" * 70)
    print()

    # Step 1: Load trajectory data
    print("Step 1: Loading trajectory data from EmbeddingDatabase...")
    print("-" * 70)
    traj_data = load_kaist_trajectories(
        dataset_name=args.dataset,
        robot_embodiment=args.robot,
    )

    rollout_ids = traj_data["rollout_ids"]
    trajectories = traj_data["trajectories"]
    n = len(rollout_ids)

    print(f"\nLoaded {n} rollouts")
    print(f"  State dimension: {traj_data['state_dim']}")
    print(f"  Action dimension: {traj_data['action_dim']}")
    print(f"  Timesteps: {traj_data['timesteps']}")
    print()

    # Step 2: Extract sub-trajectory windows
    print("Step 2: Extracting sub-trajectory windows...")
    print("-" * 70)

    window_config = WindowConfig(
        window_size=args.window_size,
        stride=args.stride,
        source=args.source,
        flatten=True,
    )

    windower = TrajectoryWindower(window_config)

    print(f"Window configuration:")
    print(f"  Window size: {window_config.window_size} timesteps")
    print(f"  Stride: {window_config.stride} timesteps")
    print(f"  Source: {window_config.source}")
    print(f"  Flatten: {window_config.flatten}")
    print()

    # Extract all windows
    all_windows, window_to_rollout, window_positions = windower.extract_all_windows(
        trajectories
    )

    # Compute statistics
    window_stats = compute_window_statistics(all_windows, window_to_rollout, n)

    print("Window extraction results:")
    print(f"  Total windows: {window_stats['total_windows']}")
    print(f"  Window dimension: {window_stats['window_dim']}")
    print(f"  Avg windows per rollout: {window_stats['avg_windows_per_rollout']:.1f}")
    print(f"  Min windows per rollout: {window_stats['min_windows_per_rollout']}")
    print(f"  Max windows per rollout: {window_stats['max_windows_per_rollout']}")
    print()

    # Step 3: Cluster windows to build coverage taxonomy
    print("Step 3: Clustering windows to build coverage taxonomy...")
    print("-" * 70)

    # L2-normalize windows before clustering
    window_norms = np.linalg.norm(all_windows, axis=1, keepdims=True)
    window_norms[window_norms == 0] = 1.0
    normalized_windows = all_windows / window_norms

    taxonomy = build_window_coverage_taxonomy(
        window_embeddings=normalized_windows,
        window_to_rollout=window_to_rollout,
        rollout_ids=rollout_ids,
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )

    A = taxonomy["A"]
    w = taxonomy["w"]
    m = taxonomy["n_clusters"]

    # Print detailed statistics
    cluster_stats = get_window_cluster_statistics(
        cluster_labels=taxonomy["cluster_labels"],
        cluster_sizes=taxonomy["cluster_sizes"],
        window_to_rollout=window_to_rollout,
        rollout_ids=rollout_ids,
        A=A,
        top_k=5,
    )
    print_window_cluster_statistics(cluster_stats)

    # Step 4: Build similarity matrix (using pooled trajectory embeddings)
    print("Step 4: Building similarity matrix (using pooled embeddings)...")
    print("-" * 70)

    # Load pooled embeddings for similarity computation
    embedding_data = load_kaist_embeddings(
        dataset_name=args.dataset,
        robot_embodiment=args.robot,
        pool_method="concat",
    )

    # Verify rollout ID ordering matches
    assert embedding_data["rollout_ids"] == rollout_ids, (
        "Rollout ID mismatch between trajectories and embeddings!"
    )

    S = build_similarity_matrix(
        embeddings=embedding_data["embeddings"],
        rollout_ids=rollout_ids,
        top_k=args.top_k,
    )

    # Validate and print statistics
    validate_similarity_matrix(S)
    sim_stats = get_similarity_statistics(S)
    print_similarity_statistics(sim_stats)

    # Step 5: Create costs (uniform for now)
    print("Step 5: Creating cost vector...")
    print("-" * 70)
    costs = np.ones(n)
    print(f"Using uniform costs: all items have cost = 1.0")
    print()

    # Step 6: Construct ProblemInstance
    print("Step 6: Constructing ProblemInstance...")
    print("-" * 70)

    metadata = {
        "dataset_name": args.dataset,
        "robot_embodiment": args.robot,
        "clustering_mode": "window-based",
        "window_size": args.window_size,
        "stride": args.stride,
        "source": args.source,
        "n_clusters": m,
        "total_windows": window_stats['total_windows'],
        "avg_windows_per_rollout": window_stats['avg_windows_per_rollout'],
        "window_dim": window_stats['window_dim'],
        "top_k": args.top_k,
        "seed": args.seed,
        "state_dim": traj_data["state_dim"],
        "action_dim": traj_data["action_dim"],
        "timesteps": traj_data["timesteps"],
    }

    instance = ProblemInstance(
        costs=costs,
        A=A,
        w=w,
        S=S,
        K=args.K,
        alpha=args.alpha,
        beta=args.beta,
        rollout_ids=rollout_ids,
        metadata=metadata,
    )

    print("ProblemInstance created successfully!")
    print()

    # Step 7: Save to disk
    print("Step 7: Saving ProblemInstance...")
    print("-" * 70)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    instance.save(output_path)
    print()

    # Step 8: Print final statistics and analysis
    print("Step 8: Final validation and analysis...")
    print("-" * 70)
    instance.print_statistics()

    # Additional validation checks
    print("\nValidation Checks:")
    print("-" * 70)
    print(f"  n = {n} rollouts")
    print(f"  m = {m} behavior clusters")
    print(f"  N_w = {window_stats['total_windows']} windows")
    print(f"  K = {args.K} target cardinality")
    print(f"  K/n ratio: {args.K/n:.2%}")
    print()

    print("Coverage Matrix A:")
    print(f"  Shape: {A.shape}")
    print(f"  Non-zeros: {A.nnz}")
    print(f"  Density: {A.nnz / (m * n) * 100:.2f}%")
    print(f"  Avg clusters per rollout: {cluster_stats['avg_clusters_per_rollout']:.2f}")
    print(f"  Min clusters per rollout: {cluster_stats['min_clusters_per_rollout']}")
    print(f"  Max clusters per rollout: {cluster_stats['max_clusters_per_rollout']}")
    print()

    print("Similarity Matrix S:")
    print(f"  Shape: {S.shape}")
    print(f"  Non-zeros: {S.nnz}")
    print(f"  Density: {S.nnz / (n * n) * 100:.2f}%")
    print()

    print("Other:")
    print(f"  Weight sum: {w.sum():.6f} (should be ~1.0)")
    print(f"  Cost sum: {costs.sum():.1f}")
    print()

    # Coverage problem characterization
    print("=" * 70)
    print("COVERAGE PROBLEM CHARACTERIZATION")
    print("=" * 70)
    print()
    print("Window-Based Clustering Results:")
    print(f"  • Clustered N_w={window_stats['total_windows']} windows → m={m} behavior clusters")
    print(f"  • Coverage matrix A: ({m}, {n}) with {A.nnz} non-zeros")
    print(f"  • Avg {cluster_stats['avg_clusters_per_rollout']:.1f} clusters per rollout (many-to-many mapping)")
    print()

    # Estimate coverage difficulty
    avg_rollouts_per_cluster = A.nnz / m
    coverage_redundancy = avg_rollouts_per_cluster / n

    print("Coverage Difficulty Analysis:")
    print(f"  • Avg rollouts per cluster: {avg_rollouts_per_cluster:.1f}")
    print(f"  • Coverage redundancy: {coverage_redundancy:.2%}")
    print(f"  • Selection rate K/n: {args.K/n:.2%}")

    if avg_rollouts_per_cluster < args.K:
        print(f"  • Assessment: Need to carefully select K={args.K} rollouts to cover all {m} clusters")
    else:
        print(f"  • Assessment: High redundancy - many rollouts share common behaviors")

    print()

    print("=" * 70)
    print("DONE! Problem instance saved to:", output_path.resolve())
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Inspect instance: python scripts/curation/inspect_instance.py", args.output)
    print("  2. Run solver: python scripts/curation/run_qubo_solver.py", args.output)
    print("  3. Compare solvers: python scripts/curation/run_batch_experiments.py")


if __name__ == "__main__":
    main()
