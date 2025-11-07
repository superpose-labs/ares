"""Problem instance representation for QUBO-based data curation.

This module defines the ProblemInstance dataclass that encapsulates all
components needed for the QUBO formulation: coverage matrix A, weights w,
similarity matrix S, and optimization parameters.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass
class ProblemInstance:
    """QUBO problem instance for data curation.

    Attributes:
        costs: np.ndarray of shape (n,) - cost per item (can be uniform)
        A: scipy.sparse matrix of shape (m, n) - coverage matrix where A[i,k]=1
           if item k covers element i
        w: np.ndarray of shape (m,) - element weights (should sum to ~1.0)
        S: scipy.sparse matrix of shape (n, n) - symmetric similarity matrix
        K: int - target cardinality (number of items to select)
        alpha: float - diversity penalty weight (default: 0.3)
        beta: float - budget penalty weight (default: 0.0)
        rollout_ids: list[str] - IDs of the rollouts in order
        metadata: dict - additional information about the problem instance
    """

    costs: np.ndarray
    A: sp.spmatrix
    w: np.ndarray
    S: sp.spmatrix
    K: int
    alpha: float = 0.3
    beta: float = 0.0
    rollout_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dimensions and data types."""
        n = len(self.costs)
        m = len(self.w)

        # Validate shapes
        assert self.A.shape == (m, n), f"A shape {self.A.shape} != ({m}, {n})"
        assert self.S.shape == (n, n), f"S shape {self.S.shape} != ({n}, {n})"
        assert len(self.rollout_ids) == n, f"rollout_ids length {len(self.rollout_ids)} != {n}"

        # Validate data types
        assert isinstance(self.costs, np.ndarray), "costs must be numpy array"
        assert isinstance(self.w, np.ndarray), "w must be numpy array"
        assert sp.issparse(self.A), "A must be sparse matrix"
        assert sp.issparse(self.S), "S must be sparse matrix"

        # Validate symmetry of S
        if not np.allclose((self.S - self.S.T).data, 0):
            raise ValueError("S must be symmetric")

        # Store dimensions in metadata
        self.metadata["n"] = n
        self.metadata["m"] = m

    def save(self, filepath: str | Path) -> None:
        """Save the problem instance to disk using pickle.

        Args:
            filepath: Path where to save the instance
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        print(f"Saved ProblemInstance to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "ProblemInstance":
        """Load a problem instance from disk.

        Args:
            filepath: Path to the saved instance

        Returns:
            Loaded ProblemInstance
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            instance = pickle.load(f)

        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")

        print(f"Loaded ProblemInstance from {filepath}")
        return instance

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics about the problem instance.

        Returns:
            Dictionary with statistics about shapes, sparsity, weights, etc.
        """
        n = self.metadata["n"]
        m = self.metadata["m"]

        stats = {
            "n_items": n,
            "n_elements": m,
            "target_cardinality": self.K,
            "alpha": self.alpha,
            "beta": self.beta,

            # Coverage matrix stats
            "A_shape": self.A.shape,
            "A_nnz": self.A.nnz,
            "A_sparsity": 1.0 - (self.A.nnz / (m * n)),
            "A_density": self.A.nnz / (m * n),

            # Similarity matrix stats
            "S_shape": self.S.shape,
            "S_nnz": self.S.nnz,
            "S_sparsity": 1.0 - (self.S.nnz / (n * n)),
            "S_density": self.S.nnz / (n * n),

            # Weight stats
            "w_sum": float(self.w.sum()),
            "w_min": float(self.w.min()),
            "w_max": float(self.w.max()),
            "w_mean": float(self.w.mean()),
            "w_std": float(self.w.std()),

            # Cost stats
            "cost_sum": float(self.costs.sum()),
            "cost_min": float(self.costs.min()),
            "cost_max": float(self.costs.max()),
            "cost_mean": float(self.costs.mean()),
            "cost_std": float(self.costs.std()),
        }

        # Add metadata
        stats.update(self.metadata)

        return stats

    def print_statistics(self) -> None:
        """Print formatted statistics about the problem instance."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("QUBO Problem Instance Statistics")
        print("=" * 60)

        print(f"\nProblem Size:")
        print(f"  Number of items (n):     {stats['n_items']}")
        print(f"  Number of elements (m):  {stats['n_elements']}")
        print(f"  Target cardinality (K):  {stats['target_cardinality']}")

        print(f"\nOptimization Parameters:")
        print(f"  Alpha (diversity):       {stats['alpha']:.3f}")
        print(f"  Beta (budget):           {stats['beta']:.3f}")

        print(f"\nCoverage Matrix A:")
        print(f"  Shape:                   {stats['A_shape']}")
        print(f"  Non-zeros:               {stats['A_nnz']}")
        print(f"  Density:                 {stats['A_density']:.4f}")
        print(f"  Sparsity:                {stats['A_sparsity']:.4f}")

        print(f"\nSimilarity Matrix S:")
        print(f"  Shape:                   {stats['S_shape']}")
        print(f"  Non-zeros:               {stats['S_nnz']}")
        print(f"  Density:                 {stats['S_density']:.4f}")
        print(f"  Sparsity:                {stats['S_sparsity']:.4f}")

        print(f"\nElement Weights w:")
        print(f"  Sum:                     {stats['w_sum']:.4f}")
        print(f"  Min:                     {stats['w_min']:.4f}")
        print(f"  Max:                     {stats['w_max']:.4f}")
        print(f"  Mean:                    {stats['w_mean']:.4f}")
        print(f"  Std:                     {stats['w_std']:.4f}")

        print(f"\nItem Costs:")
        print(f"  Sum:                     {stats['cost_sum']:.4f}")
        print(f"  Min:                     {stats['cost_min']:.4f}")
        print(f"  Max:                     {stats['cost_max']:.4f}")
        print(f"  Mean:                    {stats['cost_mean']:.4f}")
        print(f"  Std:                     {stats['cost_std']:.4f}")

        # Print additional metadata
        extra_keys = set(stats.keys()) - {
            "n_items", "n_elements", "target_cardinality", "alpha", "beta",
            "A_shape", "A_nnz", "A_sparsity", "A_density",
            "S_shape", "S_nnz", "S_sparsity", "S_density",
            "w_sum", "w_min", "w_max", "w_mean", "w_std",
            "cost_sum", "cost_min", "cost_max", "cost_mean", "cost_std",
            "n", "m"
        }

        if extra_keys:
            print(f"\nAdditional Metadata:")
            for key in sorted(extra_keys):
                print(f"  {key}: {stats[key]}")

        print("=" * 60 + "\n")
