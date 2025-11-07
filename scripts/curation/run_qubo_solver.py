#!/usr/bin/env python3
"""Run QUBO solvers on KAIST dataset curation problem.

This script loads a ProblemInstance, runs one or more solvers (greedy, CP-SAT),
evaluates their solutions, and saves results to JSON.

Usage:
    python scripts/curation/run_qubo_solver.py --K 100
    python scripts/curation/run_qubo_solver.py --K-percentage 50
    python scripts/curation/run_qubo_solver.py --solvers greedy
    python scripts/curation/run_qubo_solver.py --solvers cpsat --cpsat-time-limit 600
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from superpose_data_engine.evaluation import (
    compare_solutions,
    evaluate_random_baseline,
    evaluate_solution,
)
from superpose_data_engine.psl import ProblemInstance
from superpose_data_engine.solvers import lazy_greedy_with_diversity, solve_cpsat


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QUBO solvers for data curation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output
    parser.add_argument(
        "--instance",
        type=str,
        default="data/curation/kaist_instance.pkl",
        help="Path to ProblemInstance pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/curation",
        help="Directory to save results",
    )

    # Cardinality selection (mutually exclusive)
    cardinality_group = parser.add_mutually_exclusive_group()
    cardinality_group.add_argument(
        "--K",
        type=int,
        default=None,
        help="Target cardinality (number of items to select)",
    )
    cardinality_group.add_argument(
        "--K-percentage",
        type=float,
        default=None,
        help="Target cardinality as percentage of n (e.g., 30, 50, 70, 90)",
    )

    # Solver selection
    parser.add_argument(
        "--solvers",
        type=str,
        nargs="+",
        choices=["greedy", "cpsat", "all"],
        default=["all"],
        help="Which solvers to run",
    )

    # CP-SAT parameters
    parser.add_argument(
        "--cpsat-time-limit",
        type=int,
        default=300,
        help="CP-SAT solver time limit in seconds",
    )

    # Baseline
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include random baseline for comparison",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=10,
        help="Number of random baseline trials",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and results",
    )

    return parser.parse_args()


def numpy_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json_serializable(item) for item in obj]
    else:
        return obj


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("=" * 80)
    print("QUBO Solver for Data Curation")
    print("=" * 80)

    # Load instance
    instance_path = Path(args.instance)
    if not instance_path.exists():
        print(f"Error: Instance file not found: {instance_path}")
        sys.exit(1)

    print(f"\nLoading instance from: {instance_path}")
    instance = ProblemInstance.load(instance_path)

    # Override K if specified
    original_K = instance.K
    if args.K_percentage is not None:
        instance.K = int(len(instance.rollout_ids) * args.K_percentage / 100)
        print(f"Setting K from {args.K_percentage}% of n={len(instance.rollout_ids)}: K={instance.K}")
    elif args.K is not None:
        instance.K = args.K
        print(f"Overriding K with specified value: K={instance.K}")

    # Print instance statistics
    if args.verbose:
        instance.print_statistics()
    else:
        print(f"\nProblem size: n={instance.metadata['n']}, m={instance.metadata['m']}, K={instance.K}")
        print(f"Selection rate: {instance.K / instance.metadata['n']:.1%}")

    # Determine which solvers to run
    solvers_to_run = []
    if "all" in args.solvers:
        solvers_to_run = ["greedy", "cpsat"]
    else:
        solvers_to_run = args.solvers

    print(f"\nRunning solvers: {', '.join(solvers_to_run)}")

    # Store results
    results = {}

    # Run greedy solver
    if "greedy" in solvers_to_run:
        print("\n" + "=" * 80)
        print("Running Greedy Solver")
        print("=" * 80)

        greedy_result = lazy_greedy_with_diversity(instance, verbose=True)
        greedy_metrics = evaluate_solution(
            instance, greedy_result["selected_indices"], verbose=args.verbose
        )

        # Combine results
        results["greedy"] = {**greedy_result, **greedy_metrics}

        print(f"\nGreedy solver finished:")
        print(f"  Runtime: {greedy_result['runtime_seconds']:.2f}s")
        print(f"  Coverage: {greedy_metrics['coverage_score']:.4f} ({greedy_metrics['coverage_fraction']:.1%})")
        print(f"  Objective: {greedy_metrics['objective_value']:.4f}")

    # Run CP-SAT solver
    if "cpsat" in solvers_to_run:
        print("\n" + "=" * 80)
        print("Running CP-SAT Solver")
        print("=" * 80)

        cpsat_result = solve_cpsat(
            instance,
            time_limit_seconds=args.cpsat_time_limit,
            verbose=True,
        )

        # Only evaluate if solution found
        if cpsat_result["solve_status"] in ["optimal", "feasible"]:
            cpsat_metrics = evaluate_solution(
                instance, cpsat_result["selected_indices"], verbose=args.verbose
            )
            results["cpsat"] = {**cpsat_result, **cpsat_metrics}

            print(f"\nCP-SAT solver finished:")
            print(f"  Status: {cpsat_result['solve_status']}")
            print(f"  Runtime: {cpsat_result['runtime_seconds']:.2f}s")
            print(f"  Coverage: {cpsat_metrics['coverage_score']:.4f} ({cpsat_metrics['coverage_fraction']:.1%})")
            print(f"  Objective: {cpsat_metrics['objective_value']:.4f}")
        else:
            results["cpsat"] = cpsat_result
            print(f"\nCP-SAT solver failed: {cpsat_result['solve_status']}")

    # Run random baseline if requested
    if args.include_random:
        print("\n" + "=" * 80)
        print(f"Running Random Baseline ({args.random_trials} trials)")
        print("=" * 80)

        random_results = evaluate_random_baseline(
            instance, num_trials=args.random_trials, seed=args.seed
        )

        print(f"\nRandom baseline results:")
        print(f"  Mean coverage: {random_results['mean_coverage_score']:.4f} ± {random_results['std_coverage_score']:.4f}")
        print(f"  Mean objective: {random_results['mean_objective']:.4f} ± {random_results['std_objective']:.4f}")
        print(f"  Mean redundancy: {random_results['mean_redundancy']:.4f} ± {random_results['std_redundancy']:.4f}")

        results["random"] = random_results

    # Compare solutions
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("Comparing Solutions")
        print("=" * 80)

        # Create solutions dict for comparison (exclude random baseline)
        solutions_to_compare = {
            name: res for name, res in results.items() if name != "random"
        }

        comparison = compare_solutions(instance, solutions_to_compare, verbose=True)

        print(f"\nBest solver by objective: {comparison['best_by_objective']}")
        print(f"Best solver by coverage:  {comparison['best_by_coverage']}")

    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"kaist_selected_K{instance.K}.json"
    output_path = output_dir / output_filename

    # Convert numpy types to JSON-serializable
    json_results = {
        "instance_info": {
            "instance_path": str(instance_path),
            "n": instance.metadata["n"],
            "m": instance.metadata["m"],
            "K": instance.K,
            "alpha": instance.alpha,
            "selection_rate": instance.K / instance.metadata["n"],
        },
        "solvers": numpy_to_json_serializable(results),
    }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Also save a CSV summary
    if len(results) > 0:
        summary_data = []
        for solver_name, solver_results in results.items():
            if solver_name == "random":
                summary_data.append({
                    "solver": "random (mean)",
                    "coverage_score": solver_results["mean_coverage_score"],
                    "coverage_fraction": solver_results["mean_coverage_score"] / instance.w.sum(),
                    "objective_value": solver_results["mean_objective"],
                    "avg_redundancy": solver_results["mean_redundancy"],
                    "runtime_seconds": 0.0,
                })
            elif "selected_indices" in solver_results:
                summary_data.append({
                    "solver": solver_name,
                    "coverage_score": solver_results["coverage_score"],
                    "coverage_fraction": solver_results["coverage_fraction"],
                    "objective_value": solver_results["objective_value"],
                    "avg_redundancy": solver_results["avg_redundancy"],
                    "runtime_seconds": solver_results["runtime_seconds"],
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_dir / f"kaist_summary_K{instance.K}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("QUBO Solver Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
