# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ARES (Automatic Robot Evaluation System) is a platform for ingesting, curating, and evaluating robot data using ML models. It transforms raw robot demonstrations into structured, searchable datasets without requiring heavy infrastructure.

The system has three main components:
1. **Ingestion**: Transform raw robot data into structured format using VLMs
2. **Annotation**: Annotate rollouts with pseudo-labels and detections
3. **Curation**: Analyze data distributions and select data for training/evaluation

## Development Commands

### Setup
```bash
# Install dependencies (if not using devcontainer)
pip install -r requirements.txt
pip install -e .

# Start MongoDB for AnnotationDatabase
docker-compose -f mongo-docker-compose.yml up -d

# Setup with devcontainer (recommended)
# Use "Dev Containers: Reopen in Container" in VSCode/Cursor
```

### Running the Application
```bash
# Run Streamlit web app for data visualization
streamlit run src/ares/app/webapp.py

# Run complete ingestion pipeline (structured + embedding + grounding)
python main.py

# Run individual ingestion steps
python scripts/run_structured_ingestion.py
python scripts/run_trajectory_embedding_ingestion.py
python scripts/annotating/run_grounding.py
```

### Data Management
```bash
# Download pre-ingested data from HuggingFace Hub
chmod +x scripts/release/pull_from_hub.sh
./scripts/release/pull_from_hub.sh

# Upload data to HuggingFace Hub
python -m scripts.release.push_to_hub

# Self-heal databases (sync StructuredDatabase with AnnotationDatabase and EmbeddingDatabase)
python scripts/self_heal.py
```

### Annotation Scripts
```bash
# Generate pseudo Embodied Chain-of-Thought annotations
python scripts/annotating/run_pseudo_ecot.py

# Annotate success criteria
python scripts/annotating/run_success_criteria.py

# Run in-context learning annotations
python scripts/annotating/run_icl.py
```

### Code Quality
```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Pre-commit hooks will run black and isort automatically
```

### Evaluation
```bash
# Run VLM evaluation on Physical Intelligence demos
python scripts/eval.py
```

## Architecture

### Core Data Model: Rollout

The `Rollout` class (`src/ares/configs/base.py`) is the fundamental unit representing a robot episode/rollout. It contains:
- **Robot**: Embodiment, color, gripper type, etc.
- **Environment**: Surface, lighting, background, objects
- **Task**: Language instruction, task type, success metrics
- **Trajectory**: States, actions, observations over time

Fields with `_estimate` suffix are inferred by VLMs during ingestion. All configs inherit from `BaseConfig` which provides:
- `flatten_fields()`: Converts nested configs to flat dictionaries for SQL storage
- `get_nested_attr()`: Access nested attributes via flattened notation (e.g., `"task_language_instruction"`)

### Three-Database Architecture

1. **StructuredDatabase** (SQLite via SQLAlchemy)
   - Path: `data/robot_data.db`
   - Stores flattened `RolloutSQLModel` objects
   - Query structured metadata, performance metrics, environment details
   - Auto-creates columns when schema changes

2. **AnnotationDatabase** (MongoDB)
   - Path: `mongodb://localhost:27017`
   - Stores frame-level annotations (detections, segmentations, labels)
   - Two collections: `videos` (metadata) and `annotations` (key-value pairs)
   - Supports both whole-video and per-frame annotations

3. **EmbeddingDatabase** (FAISS)
   - Path: `data/embedding_data/`
   - Stores dense embeddings for similarity search
   - Indexes: task instructions, descriptions (via Nomic), states, actions (interpolated + normalized)
   - Index naming: `{dataset_formalname}-{robot_embodiment}-{suffix}`

### Ingestion Pipeline

The pipeline (`main.py`) runs three stages:

1. **Structured Ingestion** (`scripts/run_structured_ingestion.py`)
   - Process Open X-Embodiment datasets via TensorFlow Datasets
   - Extract frames, hard-coded metadata, and VLM-inferred fields
   - Pattern: `VLM(video, task, Field Annotations) -> Rollout -> RolloutSQLModel -> SQL DB`
   - Uses async batching for efficiency

2. **Embedding Ingestion** (`scripts/run_trajectory_embedding_ingestion.py`)
   - Embed text (task, description) with Nomic embedder
   - Embed trajectories (states, actions) after interpolation to 100 timesteps and normalization
   - Store in per-embodiment FAISS indexes

3. **Grounding Ingestion** (`scripts/annotating/run_grounding.py`)
   - Detect objects with `grounding-dino-tiny` at 5 FPS
   - Segment with `sam-vit-base`
   - Uses Modal for cloud orchestration
   - Stores in AnnotationDatabase

### Annotation Framework

The annotation system (`src/ares/annotating/`) provides:
- **AnnotatingFn**: Abstract base for annotation functions
- **orchestrate_annotating()**: Batches rollouts, manages DB connections, tracks errors/retries
- **ResultTracker**: Tracks success/failure stats during annotation
- **Modal Integration**: Cloud compute for detection/segmentation models

Annotations are composable - see Case Study 1 (ECoT) where multiple annotation types are combined.

### Frontend (Streamlit)

The webapp (`src/ares/app/webapp.py`) provides:
- **Structured Curation**: Filter by attributes (lighting, surface, success, etc.)
- **Unstructured Curation**: UMAP projections of embeddings
- **Hero Display**: View individual rollout with all metadata, video, annotations
- **Similarity Search**: Find similar rollouts by task, description, or trajectories
- **Export**: Save filtered data to CSV, Parquet, PDF, or HTML

## Key Patterns

### Pydantic Field Metadata
Field descriptions and constraints are used both for validation AND as VLM prompts:
```python
color_estimate: str = Field(
    pattern=COLOR_PATTERN,  # Used in prompts to constrain VLM output
    description="Color of the robot"
)
```

### Flattening/Reconstructing Configs
```python
rollout = Rollout(...)
flat_dict = rollout.flatten_fields()  # For SQL storage
sql_model = RolloutSQLModel(**flat_dict)

# Reconstruct from flat dict
rollout = recreate_model(flat_dict, Rollout)
```

### Custom Dataset Ingestion
See `scripts/pi_demo_ingestion.py` for example of ingesting non-OXE data. Key steps:
1. Create iterator yielding frames + metadata
2. Map to ARES format (task, success, embodiment)
3. Feed into standard ingestion pipeline

### Error Handling in Annotations
All annotation scripts use `ResultTracker` to collect errors and retry logic:
```python
tracker = ResultTracker()
try:
    result = await annotate(rollout)
    tracker.add_success(rollout_id, result)
except Exception as e:
    tracker.add_error(rollout_id, e)
```

## Important Implementation Details

- **VLM Selection**: Default is `gpt-4o` at 1 FPS for ingestion (best accuracy/cost tradeoff based on evaluation)
- **Modal Orchestration**: All detection/segmentation runs on Modal cloud compute to avoid local GPU requirements
- **Normalization**: Trajectory normalization requires full dataset (uses `NormalizationTracker` for online/batch modes)
- **Video Storage**: Videos saved to `data/videos/{dataset_filename}/{video_path}.mp4`
- **LiteLLM Integration**: Easily swap VLM providers (OpenAI, Anthropic, Gemini) via model name strings
- **Cost Optimization**: ~1 cent per rollout for VLM, <$10 for 100k frames of detection/segmentation

## Configuration Files

- `.pre-commit-config.yaml`: Runs black and isort on commit
- `pyproject.toml`: Contains black, isort, mypy, pytest config
- `mongo-docker-compose.yml`: MongoDB setup for AnnotationDatabase
- `.devcontainer/devcontainer.json`: Mounts `data/`, `/tmp/`, `.cache/huggingface/`
- `src/ares/constants.py`: Defines data paths, FPS settings, batch sizes

## Working with Open X-Embodiment Data

Datasets are defined in `src/ares/constants.py` as `DATASET_NAMES`. Each entry contains:
- `dataset_filename`: TensorFlow Datasets identifier
- `dataset_formalname`: Human-readable name used throughout ARES

Dataset-specific processing happens in `src/ares/configs/open_x_embodiment_configs.py`:
- `get_dataset_information()`: Loads metadata from `extras/oxe.csv`
- `OpenXEmbodimentEpisode`: Pydantic model with validators for data transformations

## Common Gotchas

- MongoDB must be running before using AnnotationDatabase
- Database schema auto-updates on changes to `Rollout` config
- Embedding indexes are per-embodiment - queries must use correct index name
- Frame extraction respects FPS settings in `constants.py` (1 FPS for VLM, 5 FPS for detection)
- `self_heal.py` ensures all ingested rollouts have corresponding embeddings and annotations
