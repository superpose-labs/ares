#!/usr/bin/env python3
"""
Create RLDS datasets for each percentage based on greedy solution selections.

This script reads the batch experiment results and creates filtered RLDS datasets
containing only the rollouts selected by the greedy solver for each percentage.

Usage:
    python scripts/curation/create_curated_rlds_datasets.py
    python scripts/curation/create_curated_rlds_datasets.py --results-json custom_results.json
    python scripts/curation/create_curated_rlds_datasets.py --output-base-dir data/curated_datasets/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

import tensorflow as tf
from tqdm import tqdm

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create RLDS datasets from greedy solution selections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-json",
        type=str,
        default="data/curation/batch_experiment_results.json",
        help="Path to batch experiment results JSON file",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="data/oxe/kaist_nonprehensile_converted_externally_to_rlds",
        help="Path to source RLDS dataset",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="data/curated_datasets",
        help="Base directory for output datasets",
    )
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[30, 50, 70, 90, 100],
        help="Percentages to create datasets for",
    )

    return parser.parse_args()


def load_greedy_selections(results_json_path: str, percentages: List[float]) -> dict:
    """
    Load greedy solution selections from batch experiment results.

    Returns:
        dict mapping percentage -> selected_indices (list)
    """
    print(f"Loading results from: {results_json_path}")

    with open(results_json_path, "r") as f:
        results = json.load(f)

    # Filter for greedy solutions
    greedy_results = [r for r in results if r["solver"] == "greedy"]

    print(f"Found {len(greedy_results)} greedy solutions")

    # Create mapping
    selections = {}
    for result in greedy_results:
        pct = result["percentage"]
        if pct in percentages:
            selections[pct] = result["selected_indices"]
            print(f"  Percentage {pct}%: {len(result['selected_indices'])} rollouts selected")

    # Verify all requested percentages were found
    missing = set(percentages) - set(selections.keys())
    if missing:
        raise ValueError(f"Missing percentages in results: {missing}")

    return selections


def get_tfrecord_files(dataset_path: str) -> List[Path]:
    """Get all TFRecord files in the dataset directory."""
    dataset_dir = Path(dataset_path)
    tfrecord_files = sorted(dataset_dir.glob("*.tfrecord-*"))

    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {dataset_path}")

    return tfrecord_files


def create_filtered_rlds_dataset(
    source_dataset_path: str,
    selected_indices: List[int],
    output_dir: str,
    percentage: float,
):
    """
    Create a filtered RLDS dataset containing only selected episodes.

    Args:
        source_dataset_path: Path to source TFDS dataset directory
        selected_indices: List of episode indices to include (0-based)
        output_dir: Output directory for filtered dataset
        percentage: Percentage value (for naming and metadata)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Creating dataset for {percentage}% selection")
    print(f"{'='*80}")
    print(f"Source: {source_dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Selected episodes: {len(selected_indices)}")

    # Convert to set for faster lookup
    selected_set = set(selected_indices)

    # Get all TFRecord files
    tfrecord_files = get_tfrecord_files(source_dataset_path)
    print(f"Found {len(tfrecord_files)} source TFRecord shards")

    # Create a single TFRecord dataset from all files
    file_paths = [str(f) for f in tfrecord_files]
    raw_dataset = tf.data.TFRecordDataset(file_paths)

    # Setup output sharding
    num_episodes = len(selected_indices)
    episodes_per_shard = 10  # Adjust as needed
    num_shards = (num_episodes + episodes_per_shard - 1) // episodes_per_shard

    print(f"Writing to {num_shards} output shards ({episodes_per_shard} episodes per shard)")

    # Track current shard and episode count
    current_shard = 0
    episodes_in_shard = 0
    writer = None
    written_count = 0

    # Iterate through all episodes
    print("\nProcessing episodes...")
    for episode_idx, raw_record in enumerate(tqdm(raw_dataset, desc="Filtering episodes")):
        # Check if this episode should be included
        if episode_idx not in selected_set:
            continue

        # Open new shard if needed
        if writer is None or episodes_in_shard >= episodes_per_shard:
            if writer is not None:
                writer.close()

            shard_filename = f"kaist_nonprehensile_curated_{int(percentage)}pct-train.tfrecord-{current_shard:05d}-of-{num_shards:05d}"
            shard_path = output_path / shard_filename
            writer = tf.io.TFRecordWriter(str(shard_path))

            current_shard += 1
            episodes_in_shard = 0

        # Write raw record bytes directly (preserves exact format)
        writer.write(raw_record.numpy())

        episodes_in_shard += 1
        written_count += 1

        # Early exit if we've written all selected episodes
        if written_count >= num_episodes:
            break

    # Close final writer
    if writer is not None:
        writer.close()

    print(f"\n✓ Wrote {written_count} episodes to {current_shard} shards")

    # Verify we got all episodes
    if written_count != num_episodes:
        print(f"Warning: Expected {num_episodes} episodes but wrote {written_count}")

    # Copy and update dataset_info.json
    source_info_path = Path(source_dataset_path) / "dataset_info.json"
    if source_info_path.exists():
        with open(source_info_path, "r") as f:
            dataset_info = json.load(f)

        # Update metadata
        dataset_info["name"] = f"kaist_nonprehensile_curated_{int(percentage)}pct"
        dataset_info["description"] = (
            f"KAIST Nonprehensile dataset curated to {percentage}% "
            f"({len(selected_indices)} episodes) using greedy diversity selection"
        )

        # Update split info - simplified version
        if "splits" in dataset_info:
            for split in dataset_info["splits"]:
                if split["name"] == "train":
                    # Update to reflect new shard count
                    remaining = num_episodes
                    shard_lengths = []
                    for i in range(num_shards):
                        shard_lengths.append(str(min(episodes_per_shard, remaining)))
                        remaining -= episodes_per_shard
                    split["shardLengths"] = shard_lengths
                    split["numBytes"] = "0"  # Will be computed if needed

        # Write updated info
        output_info_path = output_path / "dataset_info.json"
        with open(output_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        print(f"✓ Updated dataset_info.json")

    # Copy features.json if it exists
    source_features_path = Path(source_dataset_path) / "features.json"
    if source_features_path.exists():
        shutil.copy(source_features_path, output_path / "features.json")
        print(f"✓ Copied features.json")

    # Write selection metadata
    selection_metadata = {
        "percentage": percentage,
        "num_episodes": len(selected_indices),
        "selected_indices": sorted(selected_indices),
        "source_dataset": str(source_dataset_path),
    }

    metadata_path = output_path / "selection_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(selection_metadata, f, indent=2)

    print(f"✓ Wrote selection metadata")
    print(f"\nDataset created successfully at: {output_dir}")


def main():
    args = parse_args()

    print("=" * 80)
    print("Creating Curated RLDS Datasets from Greedy Solutions")
    print("=" * 80)

    # Verify source dataset exists
    source_path = Path(args.source_dataset)
    if not source_path.exists():
        print(f"Error: Source dataset not found: {source_path}")
        sys.exit(1)

    # Verify results file exists
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    # Load greedy selections
    try:
        selections = load_greedy_selections(args.results_json, args.percentages)
    except Exception as e:
        print(f"Error loading selections: {e}")
        sys.exit(1)

    # Create dataset for each percentage
    output_base = Path(args.output_base_dir)

    for pct in sorted(selections.keys()):
        output_dir = output_base / f"kaist_nonprehensile_curated_{int(pct)}pct" / "kaist_nonprehensile_converted_externally_to_rlds" / "0.1.0"

        try:
            create_filtered_rlds_dataset(
                source_dataset_path=str(source_path),
                selected_indices=selections[pct],
                output_dir=str(output_dir),
                percentage=pct,
            )
        except Exception as e:
            print(f"\nError creating dataset for {pct}%: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("All Datasets Created Successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {output_base}")
    print(f"Created {len(selections)} datasets:")
    for pct in sorted(selections.keys()):
        print(f"  - {int(pct)}%: {len(selections[pct])} episodes")


if __name__ == "__main__":
    main()
