"""
Export filtered KAIST dataset subset to RLDS/TFDS format.

This script takes a CSV of rollout IDs (exported from the web interface)
and creates a new RLDS dataset with only those rollouts.

Usage:
    python export_to_rlds.py --ids-csv filtered_rollouts.csv --output-dir ./exported_dataset
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sqlalchemy import select
from sqlmodel import Session

from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)
from ares.configs.base import Rollout, recreate_model
from ares.constants import ARES_VIDEO_DIR
from ares.utils.image_utils import load_video_frames


def load_rollout_ids_from_csv(csv_path: str) -> list[str]:
    """Load rollout IDs from exported CSV."""
    df = pd.read_csv(csv_path)
    if "id" not in df.columns:
        raise ValueError("CSV must contain 'id' column with rollout IDs")
    return df["id"].astype(str).tolist()


def get_rollouts_from_db(rollout_ids: list[str]) -> list[Rollout]:
    """Fetch rollouts from database by IDs."""
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    rollouts = []

    with Session(engine) as session:
        for rollout_id in rollout_ids:
            stmt = select(RolloutSQLModel).where(RolloutSQLModel.id == rollout_id)
            result = session.exec(stmt).first()
            if result:
                # Convert SQL model back to Rollout
                rollout_dict = result.model_dump()
                rollout = recreate_model(rollout_dict, Rollout)
                rollouts.append(rollout)
            else:
                print(f"Warning: Rollout {rollout_id} not found in database")

    return rollouts


def rollout_to_rlds_episode(rollout: Rollout, video_frames: np.ndarray) -> dict:
    """
    Convert ARES Rollout to RLDS episode format.

    RLDS format:
    - steps: list of {observation, action, reward, is_terminal, is_first, is_last}
    - episode_metadata: {file_path, ...}
    """
    # Parse trajectory data
    states = json.loads(rollout.trajectory.states_array) if isinstance(rollout.trajectory.states_array, str) else rollout.trajectory.states_array
    actions = json.loads(rollout.trajectory.actions_array) if isinstance(rollout.trajectory.actions_array, str) else rollout.trajectory.actions_array

    num_steps = len(states) if states else len(video_frames)

    steps = []
    for i in range(num_steps):
        # Build observation dict
        observation = {
            "image": video_frames[i] if i < len(video_frames) else video_frames[-1],
        }

        if states and i < len(states):
            observation["state"] = np.array(states[i], dtype=np.float32)

        # Build step dict
        step = {
            "observation": observation,
            "action": np.array(actions[i], dtype=np.float32) if actions and i < len(actions) else np.zeros(7, dtype=np.float32),
            "reward": 1.0 if (rollout.task.success and i == num_steps - 1) else 0.0,
            "is_terminal": i == num_steps - 1,
            "is_first": i == 0,
            "is_last": i == num_steps - 1,
            "discount": 1.0,
        }

        # Add language instruction if available
        if rollout.task.language_instruction:
            step["language_instruction"] = rollout.task.language_instruction

        steps.append(step)

    episode_metadata = {
        "file_path": rollout.path,
        "dataset_name": rollout.dataset_formalname,
        "robot_embodiment": rollout.robot.embodiment,
        "task_description": rollout.description_estimate or "",
        "success": rollout.task.success,
    }

    return {
        "steps": steps,
        "episode_metadata": episode_metadata,
    }


def export_to_rlds(rollouts: list[Rollout], output_dir: str):
    """
    Export rollouts to RLDS-compatible TFRecord format.

    Creates a directory structure:
    output_dir/
        train/
            episode_000000.tfrecord
            episode_000001.tfrecord
            ...
        dataset_info.json
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(rollouts)} rollouts to {output_dir}")

    for idx, rollout in enumerate(rollouts):
        # Load video frames
        video_path = os.path.join(ARES_VIDEO_DIR, rollout.path)
        if not os.path.exists(video_path):
            print(f"Warning: Video not found for {rollout.id}: {video_path}")
            continue

        frames = load_video_frames(video_path)

        # Convert to RLDS episode
        episode = rollout_to_rlds_episode(rollout, frames)

        # Write to TFRecord
        record_path = train_dir / f"episode_{idx:06d}.tfrecord"
        _write_episode_to_tfrecord(episode, str(record_path))

        if (idx + 1) % 10 == 0:
            print(f"  Exported {idx + 1}/{len(rollouts)} episodes")

    # Write dataset info
    dataset_info = {
        "name": "kaist_nonprehensile_filtered",
        "version": "1.0.0",
        "description": f"Filtered subset of KAIST Nonprehensile dataset ({len(rollouts)} episodes)",
        "splits": {
            "train": len(rollouts)
        },
        "features": {
            "steps": {
                "observation": {
                    "image": "uint8 array (H, W, 3)",
                    "state": "float32 array (state_dim,)"
                },
                "action": "float32 array (action_dim,)",
                "reward": "float32 scalar",
                "is_terminal": "bool",
                "is_first": "bool",
                "is_last": "bool",
                "discount": "float32 scalar",
                "language_instruction": "string"
            }
        }
    }

    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n✓ Export complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Episodes: {len(rollouts)}")
    print(f"  Format: TFRecord (RLDS-compatible)")


def _write_episode_to_tfrecord(episode: dict, filepath: str):
    """Write a single episode to TFRecord file."""
    with tf.io.TFRecordWriter(filepath) as writer:
        for step in episode["steps"]:
            # Serialize step to tf.train.Example
            feature = {}

            # Image
            image_bytes = tf.io.encode_jpeg(step["observation"]["image"]).numpy()
            feature["observation/image"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes])
            )

            # State (if present)
            if "state" in step["observation"]:
                feature["observation/state"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=step["observation"]["state"])
                )

            # Action
            feature["action"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=step["action"])
            )

            # Scalars
            feature["reward"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[step["reward"]])
            )
            feature["is_terminal"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(step["is_terminal"])])
            )
            feature["is_first"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(step["is_first"])])
            )
            feature["is_last"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(step["is_last"])])
            )
            feature["discount"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[step["discount"]])
            )

            # Language instruction
            if "language_instruction" in step:
                feature["language_instruction"] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[step["language_instruction"].encode("utf-8")]
                    )
                )

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export filtered ARES rollouts to RLDS/TFDS format"
    )
    parser.add_argument(
        "--ids-csv",
        type=str,
        required=True,
        help="Path to CSV file containing rollout IDs (exported from webapp)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for RLDS dataset",
    )

    args = parser.parse_args()

    # Load rollout IDs
    print(f"Loading rollout IDs from {args.ids_csv}")
    rollout_ids = load_rollout_ids_from_csv(args.ids_csv)
    print(f"Found {len(rollout_ids)} rollout IDs")

    # Fetch rollouts from database
    print(f"Fetching rollouts from database...")
    rollouts = get_rollouts_from_db(rollout_ids)
    print(f"Loaded {len(rollouts)} rollouts")

    # Export to RLDS
    export_to_rlds(rollouts, args.output_dir)
