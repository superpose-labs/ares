"""
Custom script to ingest only the KAIST Nonprehensile dataset.
Run this after downloading the dataset with oxe-downloader.

Usage:
    # Test with 5 episodes first
    python scripts/kaist/ingest_kaist.py --max-episodes 5

    # Run on full dataset
    python scripts/kaist/ingest_kaist.py
"""

import argparse
import asyncio
import os
import sys
import warnings
from contextlib import contextmanager

# Suppress TensorFlow cleanup warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress runtime warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

# Disable verbose LiteLLM logging completely
os.environ["LITELLM_LOG"] = "CRITICAL"  # Suppress all logging except critical errors
import logging
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)  # Suppress asyncio event loop warnings

# Suppress asyncio task exception logging
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@contextmanager
def suppress_tensorflow_cleanup():
    """Context manager to suppress TensorFlow __del__ errors after script completion."""
    import subprocess
    try:
        yield
    finally:
        # After main script completes, suppress stderr to hide TensorFlow cleanup errors
        try:
            import gc
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()

            # Redirect both stdout and stderr to suppress cleanup errors
            # This must happen as the very last action before Python exits
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)  # Redirect stderr (file descriptor 2)
            os.close(devnull_fd)
        except Exception:
            pass

from ares.annotating.orchestration import orchestrate_annotating
from ares.configs.open_x_embodiment_configs import get_dataset_information
from ares.constants import ARES_DATA_DIR, ARES_OXE_DIR
from ares.databases.annotation_database import ANNOTATION_DB_PATH
from ares.databases.embedding_database import EMBEDDING_DB_PATH
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
    setup_rollouts,
)
from ares.models.shortcuts import get_nomic_embedder
from scripts.annotating.run_grounding import GroundingModalAnnotatingFn
from scripts.run_structured_ingestion import (
    build_dataset,
    run_structured_database_ingestion,
)
from scripts.run_trajectory_embedding_ingestion import (
    run_embedding_database_ingestion_per_dataset,
)

# Dataset configuration
DATASET_FILENAME = "kaist_nonprehensile_converted_externally_to_rlds"
DATASET_FORMALNAME = "KAIST Nonprehensile Objects"
VLM_NAME = "gpt-4o"  # Using gpt-4o for much higher TPM limits (2M vs 200k for mini)


def run_ingestion_pipeline(
    ds,
    dataset_info,
    dataset_formalname,
    vlm_name,
    engine,
    dataset_filename,
    embedder,
    split,
    skip_grounding=False,
):
    """Run the three-stage ingestion pipeline."""
    print(f"\n{'='*60}")
    print(f"STAGE 1: Structured Ingestion (VLM inference)")
    print(f"{'='*60}")

    # Stage 1: Structured ingestion (VLM extracts metadata)
    structured_failures, new_rollout_ids = asyncio.run(
        run_structured_database_ingestion(
            ds,
            dataset_info,
            dataset_formalname,
            vlm_name,
            engine,
            dataset_filename,
        )
    )
    print(f"✓ Structured ingestion complete. Failures: {len(structured_failures)}")

    # Reconstitute ALL rollouts from database for this dataset
    # We'll use all rollouts for embedding and grounding (filtering happens later)
    all_rollouts = setup_rollouts(engine, dataset_formalname)

    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts found for {dataset_formalname} in {split}")

    if new_rollout_ids is not None and len(new_rollout_ids) > 0:
        print(f"ℹ️  {len(new_rollout_ids)} new rollouts ingested this run")
    else:
        print(f"ℹ️  No new rollouts (all {len(all_rollouts)} episodes already in database)")

    print(f"\n{'='*60}")
    print(f"STAGE 2: Embedding Ingestion (FAISS indexing)")
    print(f"{'='*60}")

    # Stage 2: Clear old KAIST indexes and rebuild from scratch to avoid duplicates
    from ares.databases.embedding_database import IndexManager, FaissIndex, META_INDEX_NAMES
    index_manager = IndexManager(EMBEDDING_DB_PATH, index_class=FaissIndex)

    # Delete existing KAIST-specific indexes (states, actions) to prevent duplicates
    # Note: We also delete shared indexes (task, description) as they'll be rebuilt with all datasets
    indexes_to_delete = [k for k in index_manager.indices.keys() if dataset_formalname in k or k in META_INDEX_NAMES]
    if indexes_to_delete:
        print(f"ℹ️  Deleting {len(indexes_to_delete)} existing indexes to rebuild from scratch:")
        for idx_name in indexes_to_delete:
            print(f"    - {idx_name}")
            index_manager.delete_index(idx_name)

    # Now rebuild indexes with ALL KAIST rollouts (will recreate shared indexes too)
    run_embedding_database_ingestion_per_dataset(
        all_rollouts, embedder, index_path=EMBEDDING_DB_PATH
    )
    print(f"✓ Embedding ingestion complete. Indexed {len(all_rollouts)} rollouts")

    if skip_grounding:
        print(f"\n{'='*60}")
        print(f"STAGE 3: Grounding Annotation (SKIPPED)")
        print(f"{'='*60}")
        grounding_failures = []
    else:
        print(f"\n{'='*60}")
        print(f"STAGE 3: Grounding Annotation (Modal detection/segmentation)")
        print(f"{'='*60}")

        # Check annotation database to find which rollouts need grounding
        from ares.databases.annotation_database import AnnotationDatabase, get_video_id
        ann_db = AnnotationDatabase(ANNOTATION_DB_PATH)

        rollouts_to_ground = []
        already_annotated = 0

        for r in all_rollouts:
            video_id = get_video_id(r.dataset_filename, r.filename + '.mp4')
            # Check if this rollout already has grounding annotations
            if ann_db.annotations.count_documents({'video_id': video_id}) == 0:
                rollouts_to_ground.append(r)
            else:
                already_annotated += 1

        print(f"ℹ️  Found {already_annotated} rollouts already grounded, {len(rollouts_to_ground)} need grounding")

        if len(rollouts_to_ground) == 0:
            print(f"✓ All rollouts already have grounding annotations")
            grounding_failures = []
        else:
            # Stage 3: Grounding annotation (object detection via Modal)
            annotation_results, grounding_failures = orchestrate_annotating(
                engine_path=ROBOT_DB_PATH,
                ann_db_path=ANNOTATION_DB_PATH,
                annotating_fn=GroundingModalAnnotatingFn(),
                rollout_ids=[str(r.id) for r in rollouts_to_ground],
                failures_path=os.path.join(
                    ARES_DATA_DIR,
                    "annotating_failures",
                    f"grounding_{dataset_filename}_{split}.pkl",
                ),
            )
            print(f"✓ Grounding annotation complete. Failures: {len(grounding_failures)}")

    return dict(
        structured_failures=structured_failures,
        grounding_failures=[f.__dict__ for f in grounding_failures],
    )


if __name__ == "__main__":
    with suppress_tensorflow_cleanup():
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Ingest KAIST Nonprehensile dataset into ARES"
        )
        parser.add_argument(
            "--max-episodes",
            type=int,
            default=None,
            help="Maximum number of episodes to process (default: all). Use small number like 5 for testing.",
        )
        parser.add_argument(
            "--skip-grounding",
            action="store_true",
            help="Skip grounding annotation stage (saves Modal credits for testing)",
        )
        args = parser.parse_args()

        print("="*60)
        print("KAIST Nonprehensile Dataset Ingestion")
        print("="*60)

        if args.max_episodes:
            print(f"⚠️  TEST MODE: Processing only {args.max_episodes} episodes")
        if args.skip_grounding:
            print(f"⚠️  Skipping grounding annotation stage")

        # Check if dataset exists
        dataset_path = os.path.join(ARES_OXE_DIR, DATASET_FILENAME)
        if not os.path.exists(dataset_path):
            print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
            print(f"\nPlease download it first using:")
            print(f"  oxe-download --dataset {DATASET_FILENAME} --path {ARES_OXE_DIR}")
            exit(1)

        print(f"\n✓ Found dataset at: {dataset_path}")

        # Setup databases
        print(f"\nSetting up databases...")
        print(f"  - SQLite: {ROBOT_DB_PATH}")
        print(f"  - MongoDB: {ANNOTATION_DB_PATH}")
        print(f"  - FAISS: {EMBEDDING_DB_PATH}")

        engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
        embedder = get_nomic_embedder()

        # Load dataset
        print(f"\nLoading dataset...")
        builder, dataset_dict = build_dataset(DATASET_FILENAME, ARES_OXE_DIR)
        print(f"✓ Dataset loaded with splits: {list(dataset_dict.keys())}")

        # Process each split
        for split in dataset_dict.keys():
            ds = dataset_dict[split]

            # Limit episodes if max_episodes is set
            total_episodes = len(ds)
            if args.max_episodes:
                ds = ds.take(args.max_episodes)
                print(f"\n{'#'*60}")
                print(f"Processing split: {split} ({args.max_episodes} of {total_episodes} episodes)")
                print(f"{'#'*60}")
            else:
                print(f"\n{'#'*60}")
                print(f"Processing split: {split} ({total_episodes} episodes)")
                print(f"{'#'*60}")

            dataset_info = get_dataset_information(DATASET_FILENAME)
            dataset_info["Dataset Filename"] = DATASET_FILENAME
            dataset_info["Dataset Formalname"] = DATASET_FORMALNAME
            dataset_info["Split"] = split

            failures = run_ingestion_pipeline(
                ds,
                dataset_info,
                DATASET_FORMALNAME,
                VLM_NAME,
                engine,
                DATASET_FILENAME,
                embedder,
                split,
                skip_grounding=args.skip_grounding,
            )

            print(f"\n✓ Split '{split}' complete!")
            if failures["structured_failures"]:
                print(f"  ⚠ Structured failures: {len(failures['structured_failures'])}")
            if failures["grounding_failures"]:
                print(f"  ⚠ Grounding failures: {len(failures['grounding_failures'])}")

        print(f"\n{'='*60}")
        print("✓ INGESTION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"  1. View data: streamlit run src/ares/app/webapp.py")
        print(f"  2. Filter and select subset in the web interface")
        print(f"  3. Export selected data to TFDS/RLDS format")
