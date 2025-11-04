"""
Run grounding annotation on all KAIST rollouts.
"""
import os
from ares.annotating.orchestration import orchestrate_annotating
from ares.databases.annotation_database import ANNOTATION_DB_PATH
from ares.databases.structured_database import ROBOT_DB_PATH, setup_database, RolloutSQLModel, setup_rollouts
from ares.constants import ARES_DATA_DIR
from scripts.annotating.run_grounding import GroundingModalAnnotatingFn

if __name__ == "__main__":
    print("="*60)
    print("Running Grounding Annotation on all KAIST Rollouts")
    print("="*60)

    # Load all KAIST rollouts
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    rollouts = setup_rollouts(engine, "KAIST Nonprehensile Objects")

    print(f"\nFound {len(rollouts)} KAIST rollouts in database")
    print(f"Starting grounding annotation...")

    # Run grounding annotation
    annotation_results, grounding_failures = orchestrate_annotating(
        engine_path=ROBOT_DB_PATH,
        ann_db_path=ANNOTATION_DB_PATH,
        annotating_fn=GroundingModalAnnotatingFn(),
        rollout_ids=[str(r.id) for r in rollouts],
        failures_path=os.path.join(
            ARES_DATA_DIR,
            "annotating_failures",
            f"grounding_kaist_full.pkl",
        ),
    )

    print(f"\n✓ Grounding annotation complete!")
    print(f"  Successful: {len(rollouts) - len(grounding_failures)}")
    print(f"  Failures: {len(grounding_failures)}")

    if grounding_failures:
        print(f"\nFailed rollout IDs:")
        for f in grounding_failures[:10]:
            print(f"  {f.rollout_id}: {f.error}")
