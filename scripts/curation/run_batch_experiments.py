#!/usr/bin/env python3
"""Run batch experiments for multiple K values.

This script runs random sampling, greedy, and CP-SAT solvers for multiple
selection rates (30%, 50%, 70%, 90%) and generates comparative analysis.

The random sampling baseline provides a simple comparison point to evaluate
the effectiveness of optimization-based approaches.

Usage:
    python scripts/curation/run_batch_experiments.py
    python scripts/curation/run_batch_experiments.py --percentages 20 40 60 80
    python scripts/curation/run_batch_experiments.py --output-dir results/
    python scripts/curation/run_batch_experiments.py --skip-random  # Skip random baseline
    python scripts/curation/run_batch_experiments.py --skip-cpsat   # Skip CP-SAT solver
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from superpose_data_engine.evaluation import evaluate_solution
from superpose_data_engine.psl import ProblemInstance
from superpose_data_engine.solvers import (
    lazy_greedy_with_diversity,
    random_sampling,
    solve_cpsat,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch experiments for multiple K values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--instance",
        type=str,
        default="data/curation/kaist_window_instance.pkl",
        help="Path to ProblemInstance pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/curation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[30, 50, 70, 90, 100],
        help="Selection percentages to test",
    )
    parser.add_argument(
        "--cpsat-time-limit",
        type=int,
        default=300,
        help="CP-SAT time limit per instance in seconds",
    )
    parser.add_argument(
        "--skip-cpsat",
        action="store_true",
        help="Skip CP-SAT solver (only run greedy and random)",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip random sampling baseline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def run_experiment(instance, solver_name, **solver_kwargs):
    """Run a single experiment and return results."""
    if solver_name == "greedy":
        result = lazy_greedy_with_diversity(instance, verbose=True)
    elif solver_name == "cpsat":
        result = solve_cpsat(instance, verbose=True, **solver_kwargs)
    elif solver_name == "random":
        result = random_sampling(instance, verbose=True, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    # Evaluate
    if "selected_indices" in result and len(result["selected_indices"]) > 0:
        metrics = evaluate_solution(instance, result["selected_indices"], verbose=False)
        return {**result, **metrics}
    else:
        return result


def create_comparison_plots(results_df, output_dir):
    """Create comparison plots for batch experiments."""
    output_dir = Path(output_dir)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # 1. Coverage vs K
    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in results_df["solver"].unique():
        solver_data = results_df[results_df["solver"] == solver]
        ax.plot(
            solver_data["K"],
            solver_data["coverage_score"],
            marker="o",
            label=solver,
            linewidth=2,
        )

    ax.set_xlabel("Cardinality (K)", fontsize=12)
    ax.set_ylabel("Coverage Score", fontsize=12)
    ax.set_title("Coverage Score vs Selection Size", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "coverage_vs_K.png", dpi=300)
    plt.close()

    print(f"Saved plot: {output_dir / 'coverage_vs_K.png'}")

    # 2. Objective value vs K
    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in results_df["solver"].unique():
        solver_data = results_df[results_df["solver"] == solver]
        ax.plot(
            solver_data["K"],
            solver_data["objective_value"],
            marker="o",
            label=solver,
            linewidth=2,
        )

    ax.set_xlabel("Cardinality (K)", fontsize=12)
    ax.set_ylabel("Objective Value", fontsize=12)
    ax.set_title("Objective Value vs Selection Size", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "objective_vs_K.png", dpi=300)
    plt.close()

    print(f"Saved plot: {output_dir / 'objective_vs_K.png'}")

    # 3. Coverage vs Redundancy tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in results_df["solver"].unique():
        solver_data = results_df[results_df["solver"] == solver]
        scatter = ax.scatter(
            solver_data["avg_redundancy"],
            solver_data["coverage_score"],
            s=100,
            alpha=0.7,
            label=solver,
        )

        # Annotate with K values
        for _, row in solver_data.iterrows():
            ax.annotate(
                f"K={int(row['K'])}",
                (row["avg_redundancy"], row["coverage_score"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Average Redundancy (Pairwise Similarity)", fontsize=12)
    ax.set_ylabel("Coverage Score", fontsize=12)
    ax.set_title("Coverage vs Redundancy Tradeoff", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "coverage_vs_redundancy.png", dpi=300)
    plt.close()

    print(f"Saved plot: {output_dir / 'coverage_vs_redundancy.png'}")

    # 4. Runtime comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in results_df["solver"].unique():
        solver_data = results_df[results_df["solver"] == solver]
        ax.plot(
            solver_data["K"],
            solver_data["runtime_seconds"],
            marker="o",
            label=solver,
            linewidth=2,
        )

    ax.set_xlabel("Cardinality (K)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime vs Selection Size", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "runtime_vs_K.png", dpi=300)
    plt.close()

    print(f"Saved plot: {output_dir / 'runtime_vs_K.png'}")


def main():
    args = parse_args()

    np.random.seed(args.seed)

    print("=" * 80)
    print("Batch Experiments for QUBO Data Curation")
    print("=" * 80)

    # Load instance
    instance_path = Path(args.instance)
    if not instance_path.exists():
        print(f"Error: Instance not found: {instance_path}")
        sys.exit(1)

    print(f"\nLoading instance from: {instance_path}")
    instance = ProblemInstance.load(instance_path)

    n = instance.metadata["n"]
    print(f"Problem size: n={n}, m={instance.metadata['m']}")

    # Compute K values from percentages
    K_values = [int(n * pct / 100) for pct in args.percentages]
    print(f"\nRunning experiments for K={K_values}")
    print(f"Percentages: {args.percentages}")

    # Storage for all results
    all_results = []

    # Run experiments
    for pct, K in zip(args.percentages, K_values):
        print("\n" + "=" * 80)
        print(f"Experiment: K={K} ({pct:.0f}% of n={n})")
        print("=" * 80)

        # Update instance K
        instance.K = K

        # Determine number of solvers
        num_solvers = 3
        if args.skip_cpsat:
            num_solvers -= 1
        if args.skip_random:
            num_solvers -= 1

        solver_idx = 1

        # Run random sampling baseline
        if not args.skip_random:
            print(f"\n[{solver_idx}/{num_solvers}] Running Random Sampling Baseline for K={K}")
            print("-" * 80)
            random_result = run_experiment(instance, "random", seed=args.seed)

            if "selected_indices" in random_result:
                all_results.append({
                    "solver": "random",
                    "K": K,
                    "percentage": pct,
                    **random_result,
                })
                print(f"Random: coverage={random_result['coverage_score']:.4f}, "
                      f"objective={random_result['objective_value']:.4f}, "
                      f"runtime={random_result['runtime_seconds']:.4f}s")

            solver_idx += 1

        # Run greedy
        print(f"\n[{solver_idx}/{num_solvers}] Running Greedy Solver for K={K}")
        print("-" * 80)
        greedy_result = run_experiment(instance, "greedy")

        if "selected_indices" in greedy_result:
            all_results.append({
                "solver": "greedy",
                "K": K,
                "percentage": pct,
                **greedy_result,
            })
            print(f"Greedy: coverage={greedy_result['coverage_score']:.4f}, "
                  f"objective={greedy_result['objective_value']:.4f}, "
                  f"runtime={greedy_result['runtime_seconds']:.2f}s")

        solver_idx += 1

        # Run CP-SAT
        if not args.skip_cpsat:
            print(f"\n[{solver_idx}/{num_solvers}] Running CP-SAT Solver for K={K}")
            print("-" * 80)
            cpsat_result = run_experiment(
                instance,
                "cpsat",
                time_limit_seconds=args.cpsat_time_limit,
            )

            if "selected_indices" in cpsat_result and len(cpsat_result["selected_indices"]) > 0:
                all_results.append({
                    "solver": "cpsat",
                    "K": K,
                    "percentage": pct,
                    **cpsat_result,
                })
                print(f"CP-SAT: coverage={cpsat_result['coverage_score']:.4f}, "
                      f"objective={cpsat_result['objective_value']:.4f}, "
                      f"runtime={cpsat_result['runtime_seconds']:.2f}s, "
                      f"status={cpsat_result['solve_status']}")
            else:
                print(f"CP-SAT failed: {cpsat_result.get('solve_status', 'unknown')}")

    # Create results dataframe
    if not all_results:
        print("\nNo results collected!")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)

    # Select columns for summary
    summary_cols = [
        "solver",
        "K",
        "percentage",
        "coverage_score",
        "coverage_fraction",
        "objective_value",
        "avg_redundancy",
        "max_redundancy",
        "num_covered_elements",
        "runtime_seconds",
    ]

    # Only include columns that exist
    summary_cols = [col for col in summary_cols if col in results_df.columns]
    summary_df = results_df[summary_cols]

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as CSV
    csv_path = output_dir / "batch_experiment_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {csv_path}")

    # Save full results as JSON
    json_path = output_dir / "batch_experiment_results.json"
    with open(json_path, "w") as f:
        # Convert numpy arrays to lists
        json_results = []
        for result in all_results:
            json_result = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    json_result[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    json_result[k] = float(v)
                else:
                    json_result[k] = v
            json_results.append(json_result)

        json.dump(json_results, f, indent=2)

    print(f"Full results saved to: {json_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("Summary Results")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Create comparison plots
    print(f"\n{'=' * 80}")
    print("Generating comparison plots...")
    print("=" * 80)

    try:
        create_comparison_plots(summary_df, output_dir)
        print("\nAll plots generated successfully!")
    except Exception as e:
        print(f"\nWarning: Failed to generate plots: {e}")

    # Print best results per K
    print(f"\n{'=' * 80}")
    print("Best Results per K")
    print("=" * 80)

    for K in K_values:
        K_results = summary_df[summary_df["K"] == K]
        if len(K_results) > 0:
            best_idx = K_results["objective_value"].idxmax()
            best = K_results.loc[best_idx]
            print(f"\nK={K}: Best solver = {best['solver']}")
            print(f"  Objective: {best['objective_value']:.4f}")
            print(f"  Coverage:  {best['coverage_score']:.4f} ({best['coverage_fraction']:.1%})")
            print(f"  Redundancy: {best['avg_redundancy']:.4f}")
            print(f"  Runtime:   {best['runtime_seconds']:.2f}s")

    print(f"\n{'=' * 80}")
    print("Batch Experiments Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
