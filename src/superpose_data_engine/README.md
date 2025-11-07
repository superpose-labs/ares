# Superpose Data Engine

QUBO-based framework for data curation. This module implements Week 1 of the QUBO framework for constructing problem instances from robot trajectory embeddings.

## Structure

```
src/superpose_data_engine/
├── __init__.py           # Package initialization
├── psl.py                # ProblemInstance dataclass
├── embeddings.py         # Load embeddings from EmbeddingDatabase
├── taxonomy.py           # Clustering → coverage matrix A and weights w
├── similarity.py         # Build sparse similarity matrix S
└── solvers/              # QUBO solvers (to be implemented)
    └── __init__.py
```

## Usage

### Building a QUBO Instance

```bash
# Build instance with default parameters
python scripts/curation/build_qubo_instance.py

# Build with custom parameters
python scripts/curation/build_qubo_instance.py \
    --n-clusters 20 \
    --top-k 32 \
    --K 100 \
    --alpha 0.3 \
    --output data/curation/my_instance.pkl
```

### Loading and Using an Instance

```python
from superpose_data_engine.psl import ProblemInstance

# Load instance
instance = ProblemInstance.load('data/curation/kaist_instance.pkl')

# Access components
n = instance.metadata['n']  # Number of items
m = instance.metadata['m']  # Number of elements (clusters)
K = instance.K              # Target cardinality

A = instance.A              # Coverage matrix (m × n, sparse)
w = instance.w              # Element weights (m,)
S = instance.S              # Similarity matrix (n × n, sparse, symmetric)
costs = instance.costs      # Item costs (n,)

# Get statistics
stats = instance.get_statistics()
instance.print_statistics()

# Access rollout IDs
rollout_ids = instance.rollout_ids  # List of UUIDs
```

## Problem Formulation

The QUBO instance represents the following optimization problem:

**Objective:**
- Maximize coverage of diverse elements (clusters)
- Minimize redundancy (similarity) between selected items
- Respect budget and cardinality constraints

**Components:**
- **A** (m × n): Coverage matrix where A[i,k] = 1 if item k covers element i
- **w** (m,): Element weights (IDF-style, sum to 1.0)
- **S** (n × n): Symmetric similarity matrix (top-k sparsified)
- **costs** (n,): Cost per item (currently uniform)
- **K**: Target number of items to select
- **alpha**: Diversity penalty weight (0.3 default)
- **beta**: Budget penalty weight (0.0 default)

## Implementation Details

### Coverage Taxonomy (taxonomy.py)

- Clusters embeddings using k-means
- Default k = sqrt(n) if not specified
- Each item belongs to exactly one cluster (simplified version)
- IDF weights: w_i = log(1 + n / cluster_size_i), normalized

### Similarity Matrix (similarity.py)

- Computes cosine similarity on L2-normalized embeddings
- Keeps top-k neighbors per row for sparsity
- Symmetrized: S = (S + S.T) / 2
- Clipped to [0, 1]

### Embeddings (embeddings.py)

- Loads trajectory embeddings (states and actions) from FAISS indexes
- Pools them via concatenation (default) or averaging
- Returns L2-normalized embeddings

## Example Statistics

For the KAIST dataset (200 rollouts):

```
Problem Size:
  Number of items (n):     200
  Number of elements (m):  14
  Target cardinality (K):  100

Coverage Matrix A:
  Shape:                   (14, 200)
  Non-zeros:               200
  Density:                 0.0714
  Sparsity:                0.9286

Similarity Matrix S:
  Shape:                   (200, 200)
  Non-zeros:               8198
  Density:                 0.2049
  Sparsity:                0.7951

Element Weights w:
  Sum:                     1.0000
  Min:                     0.0600
  Max:                     0.0845
```

## Next Steps

Week 2+ will add:
- QUBO solvers (simulated annealing, quantum annealing, etc.)
- Sub-trajectory windowing
- Multi-level taxonomies
- Performance evaluation
