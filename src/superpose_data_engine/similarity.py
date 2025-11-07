"""Build sparse similarity matrix from embeddings.

This module provides functions to compute pairwise cosine similarities between
embeddings and sparsify to keep only top-k neighbors.
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional


def build_similarity_matrix(
    embeddings: np.ndarray,
    rollout_ids: list[str],
    top_k: int = 32,
    batch_size: int = 100,
) -> sp.csr_matrix:
    """Build sparse symmetric similarity matrix.

    Computes pairwise cosine similarities between L2-normalized embeddings,
    then sparsifies by keeping only top-k neighbors per row. The result is
    symmetrized and clipped to [0, 1].

    Args:
        embeddings: np.ndarray of shape (n, d) - L2-normalized embeddings
        rollout_ids: list[str] - rollout IDs (for validation)
        top_k: int - number of top neighbors to keep per item (default: 32)
        batch_size: int - batch size for computing similarities (default: 100)

    Returns:
        scipy.sparse.csr_matrix of shape (n, n) - symmetric similarity matrix
    """
    n = len(embeddings)

    # Validate inputs
    assert len(rollout_ids) == n, f"rollout_ids length {len(rollout_ids)} != {n}"
    assert top_k > 0, f"top_k must be positive, got {top_k}"
    assert top_k <= n, f"top_k {top_k} > n {n}"

    print(f"Computing pairwise similarities for {n} items with top_k={top_k}...")

    # Ensure embeddings are L2-normalized
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Compute similarities in batches and keep only top-k
    row_indices = []
    col_indices = []
    data = []

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch_embeddings = embeddings[i:end_i]

        # Compute cosine similarity: batch_embeddings @ embeddings.T
        # Shape: (batch_size, n)
        batch_similarities = batch_embeddings @ embeddings.T

        # For each row in the batch, keep top-k neighbors
        for local_idx, global_idx in enumerate(range(i, end_i)):
            sims = batch_similarities[local_idx]

            # Get top-k indices (including self)
            # Use argpartition for efficiency
            if top_k < n:
                top_indices = np.argpartition(sims, -top_k)[-top_k:]
                top_sims = sims[top_indices]
            else:
                top_indices = np.arange(n)
                top_sims = sims

            # Store non-zero entries
            for idx, sim in zip(top_indices, top_sims):
                row_indices.append(global_idx)
                col_indices.append(idx)
                data.append(sim)

        if (i // batch_size + 1) % 5 == 0 or end_i == n:
            print(f"  Processed {end_i}/{n} rows...")

    # Build sparse matrix
    S = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    print(f"Initial similarity matrix: {S.shape}, nnz={S.nnz}")

    # Symmetrize: S = (S + S.T) / 2
    S = (S + S.T) / 2

    # Clip to [0, 1]
    S.data = np.clip(S.data, 0, 1)

    # Eliminate explicit zeros
    S.eliminate_zeros()

    print(f"Final symmetric similarity matrix: {S.shape}, nnz={S.nnz}")
    print(f"Density: {S.nnz / (n * n):.6f}")
    print(f"Sparsity: {1.0 - S.nnz / (n * n):.6f}")

    # Validate symmetry
    diff = (S - S.T)
    if diff.nnz > 0:
        max_diff = np.abs(diff.data).max()
        print(f"Warning: Matrix not perfectly symmetric, max diff: {max_diff:.2e}")

    return S


def get_similarity_statistics(S: sp.csr_matrix) -> dict:
    """Get statistics about the similarity matrix.

    Args:
        S: scipy.sparse matrix - similarity matrix

    Returns:
        Dictionary with statistics
    """
    n = S.shape[0]

    # Get non-zero values
    values = S.data

    # Compute row-wise statistics
    row_nnz = np.diff(S.indptr)  # Number of non-zeros per row

    stats = {
        "shape": S.shape,
        "nnz": S.nnz,
        "density": S.nnz / (n * n),
        "sparsity": 1.0 - S.nnz / (n * n),

        # Value statistics
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "value_mean": float(values.mean()),
        "value_median": float(np.median(values)),
        "value_std": float(values.std()),

        # Row statistics
        "row_nnz_min": int(row_nnz.min()),
        "row_nnz_max": int(row_nnz.max()),
        "row_nnz_mean": float(row_nnz.mean()),
        "row_nnz_median": float(np.median(row_nnz)),
        "row_nnz_std": float(row_nnz.std()),

        # Symmetry check
        "is_symmetric": np.allclose((S - S.T).data, 0),
    }

    return stats


def print_similarity_statistics(stats: dict) -> None:
    """Print formatted similarity matrix statistics.

    Args:
        stats: Dictionary from get_similarity_statistics()
    """
    print("\n" + "=" * 60)
    print("Similarity Matrix Statistics")
    print("=" * 60)

    print(f"\nShape and Sparsity:")
    print(f"  Shape:        {stats['shape']}")
    print(f"  Non-zeros:    {stats['nnz']}")
    print(f"  Density:      {stats['density']:.6f}")
    print(f"  Sparsity:     {stats['sparsity']:.6f}")

    print(f"\nSimilarity Values:")
    print(f"  Min:          {stats['value_min']:.4f}")
    print(f"  Max:          {stats['value_max']:.4f}")
    print(f"  Mean:         {stats['value_mean']:.4f}")
    print(f"  Median:       {stats['value_median']:.4f}")
    print(f"  Std:          {stats['value_std']:.4f}")

    print(f"\nRow Non-zeros (neighbors per item):")
    print(f"  Min:          {stats['row_nnz_min']}")
    print(f"  Max:          {stats['row_nnz_max']}")
    print(f"  Mean:         {stats['row_nnz_mean']:.1f}")
    print(f"  Median:       {stats['row_nnz_median']:.1f}")
    print(f"  Std:          {stats['row_nnz_std']:.1f}")

    print(f"\nProperties:")
    print(f"  Symmetric:    {stats['is_symmetric']}")

    print("=" * 60 + "\n")


def validate_similarity_matrix(S: sp.csr_matrix, tol: float = 1e-6) -> bool:
    """Validate that similarity matrix meets requirements.

    Args:
        S: scipy.sparse matrix - similarity matrix
        tol: float - tolerance for checks

    Returns:
        bool - True if all checks pass

    Raises:
        ValueError if any check fails
    """
    n = S.shape[0]

    # Check square
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square, got shape {S.shape}")

    # Check symmetric
    diff = (S - S.T)
    if diff.nnz > 0:
        max_diff = np.abs(diff.data).max()
        if max_diff > tol:
            raise ValueError(f"S not symmetric, max diff: {max_diff:.2e}")

    # Check values in [0, 1]
    if S.data.min() < -tol:
        raise ValueError(f"S has negative values: min={S.data.min()}")
    if S.data.max() > 1.0 + tol:
        raise ValueError(f"S has values > 1: max={S.data.max()}")

    # Check diagonal (should be all 1s if present)
    diag = S.diagonal()
    non_zero_diag = diag[diag != 0]
    if len(non_zero_diag) > 0:
        if not np.allclose(non_zero_diag, 1.0, atol=tol):
            print(f"Warning: Diagonal values not all 1.0: "
                  f"min={non_zero_diag.min():.4f}, max={non_zero_diag.max():.4f}")

    print("Similarity matrix validation passed!")
    return True
