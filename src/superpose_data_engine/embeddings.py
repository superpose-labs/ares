"""Load embeddings from ARES EmbeddingDatabase.

This module provides functions to load trajectory embeddings from FAISS indexes
and pool them appropriately for use in QUBO problem construction.
"""

import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sqlmodel import create_engine, Session, select

from ares.configs.pydantic_sql_helpers import create_flattened_model
from ares.configs.base import Rollout
from ares.constants import ARES_DATA_DIR

ROBOT_DB_PATH = f"sqlite:///{os.path.join(ARES_DATA_DIR, 'robot_data.db')}"
EMBEDDING_DB_PATH = os.path.join(ARES_DATA_DIR, "embedding_data")

# Create RolloutSQLModel for querying
RolloutSQLModel = create_flattened_model(Rollout)


def load_kaist_embeddings(
    dataset_name: str = "KAIST Nonprehensile Objects",
    robot_embodiment: str = "Franka",
    pool_method: str = "concat",
) -> dict:
    """Load KAIST embeddings from the EmbeddingDatabase.

    This function queries the StructuredDatabase for KAIST rollout IDs,
    then loads their trajectory embeddings (states and actions) from FAISS
    indexes and pools them into a single embedding vector per rollout.

    Args:
        dataset_name: Formal name of the dataset (default: "KAIST Nonprehensile Objects")
        robot_embodiment: Robot embodiment (default: "Franka")
        pool_method: How to pool states/actions embeddings:
            - "concat": Concatenate states and actions (default)
            - "mean": Average states and actions
            - "states_only": Use only states
            - "actions_only": Use only actions

    Returns:
        Dictionary with:
            - rollout_ids: list[str] - UUIDs of rollouts
            - embeddings: np.ndarray of shape (n, d) - pooled embeddings (L2-normalized)
            - embedding_dim: int - dimensionality of embeddings
            - states_dim: int - dimensionality of state embeddings
            - actions_dim: int - dimensionality of action embeddings
            - n_rollouts: int - number of rollouts
            - pool_method: str - pooling method used
    """
    # Query StructuredDatabase for KAIST rollout IDs
    engine = create_engine(ROBOT_DB_PATH)

    with Session(engine) as session:
        query = select(RolloutSQLModel).where(
            RolloutSQLModel.dataset_formalname == dataset_name
        )
        rows = session.exec(query).all()

        if not rows:
            raise ValueError(f"No rollouts found for dataset: {dataset_name}")

        rollout_ids = [str(row.id) for row in rows]

    print(f"Found {len(rollout_ids)} rollouts for {dataset_name}")

    # Load FAISS indexes
    states_index_name = f"{dataset_name}-{robot_embodiment}-states"
    actions_index_name = f"{dataset_name}-{robot_embodiment}-actions"

    states_index_path = Path(EMBEDDING_DB_PATH) / f"{states_index_name}.index"
    actions_index_path = Path(EMBEDDING_DB_PATH) / f"{actions_index_name}.index"

    if not states_index_path.exists():
        raise FileNotFoundError(f"States index not found: {states_index_path}")
    if not actions_index_path.exists():
        raise FileNotFoundError(f"Actions index not found: {actions_index_path}")

    # Load FAISS indexes
    states_index = faiss.read_index(str(states_index_path))
    actions_index = faiss.read_index(str(actions_index_path))

    # Load metadata to get ID mappings
    import json
    states_meta_path = Path(EMBEDDING_DB_PATH) / f"{states_index_name}_meta.json"
    actions_meta_path = Path(EMBEDDING_DB_PATH) / f"{actions_index_name}_meta.json"

    with open(states_meta_path) as f:
        states_meta = json.load(f)
    with open(actions_meta_path) as f:
        actions_meta = json.load(f)

    # Get dimensions
    states_feature_dim = states_meta["feature_dim"]
    states_time_steps = states_meta["time_steps"]
    actions_feature_dim = actions_meta["feature_dim"]
    actions_time_steps = actions_meta["time_steps"]

    states_total_dim = states_feature_dim * states_time_steps
    actions_total_dim = actions_feature_dim * actions_time_steps

    print(f"States: {states_feature_dim} features × {states_time_steps} timesteps = {states_total_dim} dim")
    print(f"Actions: {actions_feature_dim} features × {actions_time_steps} timesteps = {actions_total_dim} dim")

    # Create reverse ID mappings (string ID -> internal ID)
    states_id_map = {v: int(k) for k, v in states_meta["id_map"].items()}
    actions_id_map = {v: int(k) for k, v in actions_meta["id_map"].items()}

    # Extract embeddings for each rollout
    states_embeddings = []
    actions_embeddings = []
    valid_rollout_ids = []

    for rollout_id in rollout_ids:
        if rollout_id in states_id_map and rollout_id in actions_id_map:
            states_internal_id = states_id_map[rollout_id]
            actions_internal_id = actions_id_map[rollout_id]

            # Reconstruct from FAISS
            states_vec = states_index.reconstruct(states_internal_id)
            actions_vec = actions_index.reconstruct(actions_internal_id)

            states_embeddings.append(states_vec)
            actions_embeddings.append(actions_vec)
            valid_rollout_ids.append(rollout_id)
        else:
            print(f"Warning: Rollout {rollout_id} not found in embedding indexes")

    if not valid_rollout_ids:
        raise ValueError("No valid rollout embeddings found")

    states_embeddings = np.array(states_embeddings)  # (n, states_total_dim)
    actions_embeddings = np.array(actions_embeddings)  # (n, actions_total_dim)

    print(f"Loaded embeddings for {len(valid_rollout_ids)} rollouts")

    # Pool embeddings based on method
    if pool_method == "concat":
        # Concatenate states and actions
        pooled_embeddings = np.concatenate(
            [states_embeddings, actions_embeddings], axis=1
        )
        embedding_dim = states_total_dim + actions_total_dim
    elif pool_method == "mean":
        # Average states and actions (requires normalizing dimensions first)
        # Normalize each to unit norm, then average
        states_normed = states_embeddings / np.linalg.norm(
            states_embeddings, axis=1, keepdims=True
        )
        actions_normed = actions_embeddings / np.linalg.norm(
            actions_embeddings, axis=1, keepdims=True
        )
        pooled_embeddings = (states_normed + actions_normed) / 2
        embedding_dim = states_total_dim  # Assuming they're the same after normalization
    elif pool_method == "states_only":
        pooled_embeddings = states_embeddings
        embedding_dim = states_total_dim
    elif pool_method == "actions_only":
        pooled_embeddings = actions_embeddings
        embedding_dim = actions_total_dim
    else:
        raise ValueError(f"Unknown pool_method: {pool_method}")

    # L2-normalize the pooled embeddings
    norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    pooled_embeddings = pooled_embeddings / norms

    print(f"Pooled embeddings shape: {pooled_embeddings.shape}")
    print(f"Pool method: {pool_method}")

    return {
        "rollout_ids": valid_rollout_ids,
        "embeddings": pooled_embeddings,
        "embedding_dim": embedding_dim,
        "states_dim": states_total_dim,
        "actions_dim": actions_total_dim,
        "n_rollouts": len(valid_rollout_ids),
        "pool_method": pool_method,
        "dataset_name": dataset_name,
        "robot_embodiment": robot_embodiment,
    }


def load_kaist_trajectories(
    dataset_name: str = "KAIST Nonprehensile Objects",
    robot_embodiment: str = "Franka",
) -> dict:
    """Load KAIST trajectory data for window-based clustering.

    Unlike load_kaist_embeddings which pools trajectories into single vectors,
    this function returns the full trajectory data as (timesteps, features) arrays.

    Args:
        dataset_name: Formal name of the dataset
        robot_embodiment: Robot embodiment

    Returns:
        Dictionary with:
            - rollout_ids: list[str] - UUIDs of rollouts
            - trajectories: dict[str, tuple[np.ndarray, np.ndarray]]
                Mapping from rollout_id → (state_traj, action_traj)
                where state_traj.shape = (timesteps, state_dim)
                      action_traj.shape = (timesteps, action_dim)
            - state_dim: int - number of state features
            - action_dim: int - number of action features
            - timesteps: int - number of timesteps (after interpolation)
            - n_rollouts: int - number of rollouts
    """
    # Query StructuredDatabase for KAIST rollout IDs
    engine = create_engine(ROBOT_DB_PATH)

    with Session(engine) as session:
        query = select(RolloutSQLModel).where(
            RolloutSQLModel.dataset_formalname == dataset_name
        )
        rows = session.exec(query).all()

        if not rows:
            raise ValueError(f"No rollouts found for dataset: {dataset_name}")

        rollout_ids = [str(row.id) for row in rows]

    print(f"Found {len(rollout_ids)} rollouts for {dataset_name}")

    # Load FAISS indexes
    states_index_name = f"{dataset_name}-{robot_embodiment}-states"
    actions_index_name = f"{dataset_name}-{robot_embodiment}-actions"

    states_index_path = Path(EMBEDDING_DB_PATH) / f"{states_index_name}.index"
    actions_index_path = Path(EMBEDDING_DB_PATH) / f"{actions_index_name}.index"

    if not states_index_path.exists():
        raise FileNotFoundError(f"States index not found: {states_index_path}")
    if not actions_index_path.exists():
        raise FileNotFoundError(f"Actions index not found: {actions_index_path}")

    # Load FAISS indexes
    states_index = faiss.read_index(str(states_index_path))
    actions_index = faiss.read_index(str(actions_index_path))

    # Load metadata
    import json
    states_meta_path = Path(EMBEDDING_DB_PATH) / f"{states_index_name}_meta.json"
    actions_meta_path = Path(EMBEDDING_DB_PATH) / f"{actions_index_name}_meta.json"

    with open(states_meta_path) as f:
        states_meta = json.load(f)
    with open(actions_meta_path) as f:
        actions_meta = json.load(f)

    # Get dimensions
    state_dim = states_meta["feature_dim"]
    timesteps = states_meta["time_steps"]
    action_dim = actions_meta["feature_dim"]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Timesteps: {timesteps}")

    # Create reverse ID mappings
    states_id_map = {v: int(k) for k, v in states_meta["id_map"].items()}
    actions_id_map = {v: int(k) for k, v in actions_meta["id_map"].items()}

    # Extract trajectories for each rollout
    trajectories = {}
    valid_rollout_ids = []

    for rollout_id in rollout_ids:
        if rollout_id in states_id_map and rollout_id in actions_id_map:
            states_internal_id = states_id_map[rollout_id]
            actions_internal_id = actions_id_map[rollout_id]

            # Reconstruct flattened vectors
            states_flat = states_index.reconstruct(states_internal_id)
            actions_flat = actions_index.reconstruct(actions_internal_id)

            # Reshape to (timesteps, features)
            state_traj = states_flat.reshape(timesteps, state_dim)
            action_traj = actions_flat.reshape(timesteps, action_dim)

            trajectories[rollout_id] = (state_traj, action_traj)
            valid_rollout_ids.append(rollout_id)
        else:
            print(f"Warning: Rollout {rollout_id} not found in embedding indexes")

    if not valid_rollout_ids:
        raise ValueError("No valid rollout trajectories found")

    print(f"Loaded trajectories for {len(valid_rollout_ids)} rollouts")

    return {
        "rollout_ids": valid_rollout_ids,
        "trajectories": trajectories,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "timesteps": timesteps,
        "n_rollouts": len(valid_rollout_ids),
        "dataset_name": dataset_name,
        "robot_embodiment": robot_embodiment,
    }
