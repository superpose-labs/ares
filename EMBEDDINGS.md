# Embedding

##1️⃣ Task Language Instruction Embedding

- Input: Short task description string (e.g., "push the spam cam to the left top corner")
- Output: Single 768-dimensional vector
- Model: nomic-ai/nomic-embed-text-v1
- Purpose: Find rollouts with similar task instructions

##2️⃣ Description Estimate Embedding

- Input: Long description of the rollout (e.g., detailed explanation of what's happening)
- Output: Single 768-dimensional vector
- Model: nomic-ai/nomic-embed-text-v1
- Purpose: Find rollouts with similar overall descriptions

## state and action embedding:
One embedding per trajectory type per rollout:
  - 1 embedding for the entire state trajectory
  - 1 embedding for the entire action trajectory

How it works:
  - Each trajectory (e.g., states or actions) is interpolated to 100 timesteps (from whatever original length)
  - Then normalized using dataset-wide statistics (mean/std per feature dimension)
  - Then flattened into a single vector of length: 100 × feature_dim

For your KAIST dataset specifically:
  - Actions: 20-dimensional features → stored as vectors of length 2000 (100 × 20)
  - States: 21-dimensional features → stored as vectors of length 2100 (100 × 21)

### Example:
Original Data (Variable Length)

Your KAIST trajectories have widely varying lengths:
- Shortest: 22 timesteps
- Longest: 2,041 timesteps
- Average: 161.4 timesteps

For example:
- Rollout 1: 49 timesteps (states and actions at each step)
- Rollout 2: 535 timesteps (states and actions at each step)
- Rollout 4: 254 timesteps (states and actions at each step)

After Interpolation (Fixed Length for FAISS)

All trajectories are resampled to exactly 100 timesteps using linear interpolation:
- Rollout 1: 49 steps → interpolated to 100 steps
- Rollout 2: 535 steps → downsampled to 100 steps
- Rollout 4: 254 steps → downsampled to 100 steps

## Complete Embeddings Per Rollout

Each rollout in KAIST has 4 total embeddings:

| Type       | Count | Details                                            |
|------------|-------|----------------------------------------------------|
| Trajectory | 2     | States (2100-dim) + Actions (2000-dim)             |
| Language   | 2     | Task instruction (768-dim) + Description (768-dim) |
| Total      | 4     | All stored as separate vectors in FAISS            |

