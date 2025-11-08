# Complete Guide: Open X-Embodiment Dataset Workflow

This guide walks you through downloading, ingesting, visualizing, filtering, and exporting any Open X-Embodiment (OXE) dataset using ARES on a Mac.

## Prerequisites

- Mac computer with Python 3.10+
- Homebrew installed (for MongoDB)
- Git repository already cloned
- **Required API credentials:**
  - OpenAI API key (for GPT-4o in Stage 1)
  - Modal account + token (for grounding models in Stage 3)
  - Note: Nomic embedder in Stage 2 runs locally - no API key needed

---

## Step 1: Install Dependencies

### 1.1 Install Python Requirements
```bash
pip install -r requirements.txt
pip install -e .
```

### 1.2 Install MongoDB

**Option A: Using Homebrew (Recommended)**
```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community@7.0

# Start MongoDB service (will auto-restart on reboot)
brew services start mongodb-community@7.0

# Verify it's running
brew services list | grep mongodb
# Should show: mongodb-community started
```

**Option B: Using Docker**
```bash
# Start MongoDB container
docker-compose -f mongo-docker-compose.yml up -d

# Verify it's running
docker ps | grep mongo
```

### 1.3 Verify MongoDB Connection
```bash
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017'); print('✓ MongoDB connected!'); client.close()"
```

### 1.4 Install oxe-downloader
```bash
pip install oxe-downloader
```

---

## Step 2: Set Up Environment

### 2.1 Set API Keys

**Required Credentials:**

1. **OpenAI API Key** (Required for Step 1: Structured Ingestion)
   - Used for GPT-4o to extract metadata from videos
   - Cost: ~$0.01 per rollout

2. **Modal Token** (Required for Step 3: Grounding Annotation)
   - Used for cloud-based object detection and segmentation
   - Free tier: $30/month in credits (sufficient for most use cases)
   - Setup: `modal token new` (opens browser to authenticate)

3. **Nomic Embedder** (Step 2: Embedding Ingestion)
   - No API key needed - runs locally on your CPU
   - Model auto-downloads from HuggingFace

**Setup OpenAI API Key:**
Create a `.env` file or export environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Add to your `~/.zshrc` or `~/.bash_profile` to persist:
```bash
echo 'export OPENAI_API_KEY="your-openai-api-key"' >> ~/.zshrc
source ~/.zshrc
```

**Setup Modal (for grounding annotations):**
```bash
pip install modal
modal token new  # Opens browser to authenticate with Modal
```

### 2.2 Create Data Directory
```bash
mkdir -p data/oxe data/videos data/annotating_failures
```

### 2.3 Enable Your Dataset

Edit `src/ares/constants.py` to add your dataset to `DATASET_NAMES`:

```python
DATASET_NAMES = [
    {
        "dataset_filename": "your_dataset_name",  # TensorFlow Datasets identifier
        "dataset_formalname": "Your Dataset Display Name"
    },
    # ... other datasets
]
```

---

## Models Used in the Pipeline

| Stage | Model | Purpose | Runs On | API Key Needed? | Cost |
|-------|-------|---------|---------|-----------------|------|
| Stage 1 | `gpt-4o` (OpenAI) | Extract metadata from videos | OpenAI API | ✅ Yes | ~$0.01/rollout |
| Stage 2 | `nomic-embed-text-v1` | Embed text & trajectories | Local CPU | ❌ No | Free |
| Stage 3 | `grounding-dino-tiny` + `sam-vit-base` | Object detection & segmentation | Modal Cloud | ✅ Modal token | Free with credits |

**Implementation Details:**
- **Nomic Embedder** (`src/ares/models/base.py:271-285`): Downloads from HuggingFace via `sentence-transformers` library, runs inference locally
- **Grounding Models** (`src/ares/models/grounding.py`): Deployed to Modal serverless compute to avoid local GPU requirements

---

## Step 3: Download OXE Dataset

```bash
# Download your chosen dataset (replace with your dataset name)
oxe_download --dataset YOUR_DATASET_NAME --path ./data/oxe

# Examples:
# oxe_download --dataset kaist_nonprehensile_converted_externally_to_rlds --path ./data/oxe
# oxe_download --dataset cmu_play_fusion --path ./data/oxe
# oxe_download --dataset bridge --path ./data/oxe

# This will create: ./data/oxe/YOUR_DATASET_NAME/
```

**Find available datasets:** Visit [Open X-Embodiment](https://robotics-transformer-x.github.io/) or use `oxe_download --list` to see all available datasets.

**Note:** Download time depends on your internet connection and dataset size. The dataset will be in TensorFlow Datasets format.

---

## Step 4: Ingest Dataset into ARES

### 4.1 Using the General Ingestion Script

ARES provides a general-purpose ingestion script that works with any Open X-Embodiment dataset:

```bash
# Run the complete 3-stage ingestion pipeline
python scripts/ingest_oxe_dataset.py \
    --dataset-filename YOUR_DATASET_FILENAME \
    --dataset-formalname "Your Dataset Display Name"
```

**Alternative:** You can also use individual scripts for more control:
- Stage 1: `python scripts/run_structured_ingestion.py`
- Stage 2: `python scripts/run_trajectory_embedding_ingestion.py`
- Stage 3: `python scripts/annotating/run_grounding.py`

Or use the main pipeline: `python main.py`

### 4.2 Test on Small Sample First (Recommended!)

**Test with just 5 episodes to verify everything works before spending tokens:**

```bash
# Process only 5 episodes, skip expensive grounding annotation
python scripts/ingest_oxe_dataset.py \
    --dataset-filename YOUR_DATASET_FILENAME \
    --dataset-formalname "Your Dataset Display Name" \
    --max-episodes 5 \
    --skip-grounding

# Or test with grounding included (uses Modal credits)
python scripts/ingest_oxe_dataset.py \
    --dataset-filename YOUR_DATASET_FILENAME \
    --dataset-formalname "Your Dataset Display Name" \
    --max-episodes 5
```

**Expected cost for 5 episodes:** ~$0.05-0.10

Once verified, run on the full dataset:

```bash
python scripts/ingest_oxe_dataset.py \
    --dataset-filename YOUR_DATASET_FILENAME \
    --dataset-formalname "Your Dataset Display Name"
```

### 4.3 Ingestion Pipeline Stages

This script performs three stages:

### Stage 1: Structured Ingestion (15-30 min for typical dataset)
- Extracts frames from videos
- Sends frames to GPT-4o to extract metadata (task, environment, success, etc.)
- Stores structured data in SQLite database (`data/robot_data.db`)
- **Cost:** ~$0.01 per rollout (1 FPS)

### Stage 2: Embedding Ingestion (5-10 min)
- **Model:** `nomic-embed-text-v1` (Nomic AI)
- **Where it runs:** Locally on your CPU via `sentence-transformers` library
- **What it does:**
  - Embeds task instructions and descriptions into dense vectors
  - Embeds state and action trajectories for similarity search
  - Creates FAISS indexes stored in `data/embedding_data/`
- **Cost:** Free (no API key needed, model auto-downloads from HuggingFace)

### Stage 3: Grounding Annotation (10-20 min)
- **Models:**
  - Detection: `IDEA-Research/grounding-dino-tiny` (zero-shot object detection)
  - Segmentation: `facebook/sam-vit-base` (Segment Anything Model)
- **Where it runs:** Modal serverless cloud compute (avoids need for local GPU)
- **What it does:**
  - Detects objects in frames at 5 FPS using grounding-dino-tiny
  - Generates segmentation masks using SAM
  - Stores frame-level annotations in MongoDB
- **Cost:** Free with Modal's $30/month credits (typically <$10 for 100k frames)
- **Requires:** Modal account and authentication token (see Step 2.1)

**Progress Tracking:**
The script shows progress bars and status for each stage. You can interrupt with Ctrl+C and resume later - it will skip already-processed rollouts.

**Troubleshooting:**
- If MongoDB connection fails: Check that MongoDB is running with `brew services list`
- If API rate limits hit: The script automatically retries with exponential backoff
- If Modal fails: You may need to set up a Modal account and run `modal token new`

---

## Step 5: Visualize Data in Web Interface

Once ingestion completes, start the Streamlit webapp:

```bash
streamlit run src/ares/app/webapp.py
```

This opens a browser at `http://localhost:8501` with the ARES dashboard.

### Navigation:

1. **Loading Data Section (Top)**
   - Select your dataset from the dropdown
   - Data loads automatically

2. **Structured Data Filters**
   - Filter by environment (lighting, surface, background)
   - Filter by robot attributes (embodiment, gripper type)
   - Filter by task (success/failure, task type)
   - Filter by performance metrics

3. **Embedding Data Filters (UMAP Projections)**
   - View 2D UMAP projections of task instructions or descriptions
   - Click points to select clusters
   - Find semantically similar rollouts

4. **Hero Display**
   - Click any rollout to see detailed view
   - Watch video with annotations overlaid
   - See all metadata, object detections, segmentations
   - Find similar rollouts by task, description, or trajectories

5. **Data Distributions**
   - View histograms and statistics
   - Understand dataset composition

---


---

## Step 6: Build QUBO Coefficients for Data Curation

After ingesting and visualizing your data, you can use QUBO-based optimization to automatically curate a diverse, high-quality subset for training.

**What this does:**
- Extracts sub-trajectory windows from your dataset
- Clusters windows to identify distinct behavior patterns
- Builds coverage and similarity matrices for optimization
- Creates a QUBO problem instance that balances coverage and diversity

```bash
# Build QUBO instance with window-based clustering
python scripts/curation/build_window_qubo_instance.py
```

**Output:** `data/curation/kaist_window_instance.pkl`

**Customization (optional):**
```bash
python scripts/curation/build_window_qubo_instance.py \
    --window-size 10 \      # Window length in timesteps
    --stride 5 \            # Stride for sliding windows
    --n-clusters 20 \       # Number of behavior clusters
    --similarity-topL 32    # Top-L neighbors for similarity matrix
```

**Verify the instance (optional):**
```bash
python scripts/curation/inspect_instance.py data/curation/kaist_window_instance.pkl
```

This prints detailed statistics about the coverage matrix, similarity matrix, and element weights.

---

## Step 7: Run Batch Experiments to Select Data Subset

This step solves the QUBO problem using multiple solvers (random baseline, greedy, CP-SAT) and compares their performance across different selection sizes.

**What this does:**
- Runs multiple solvers for K = 30%, 50%, 70%, 90% of dataset
- Balances coverage (diverse behaviors) vs diversity (avoid redundancy)
- Generates comparison plots and summary tables
- Saves solution files for each K value

```bash
# Run all solvers with default settings
python scripts/curation/run_batch_experiments.py
```

**Customization (optional):**
```bash
python scripts/curation/run_batch_experiments.py \
    --instance data/curation/kaist_window_instance.pkl \
    --percentages 30 50 70 90 \    # Selection percentages
    --alpha 0.3 \                   # Diversity penalty weight
    --skip-random                   # Skip random baseline (faster)
```

**Output files in `data/curation/`:**
- `kaist_selected_K{60,100,140,180}.json` - Solution files for each K
- `kaist_summary_K*.csv` - Comparison tables
- `batch_experiment_results.{json,csv}` - All results
- `coverage_vs_K.png`, `objective_vs_K.png`, etc. - Visualizations

**Recommended solver:** Greedy (best quality) or CP-SAT (fastest for large datasets)

---

## Step 8: Create Curated RLDS Datasets

Finally, export the selected rollouts to RLDS format for downstream training.

**What this does:**
- Reads experiment results from Step 7
- Queries StructuredDatabase for selected rollout IDs
- Exports curated subsets to RLDS format
- Creates separate datasets for each solver and K value

```bash
# Create curated RLDS datasets from experiment results
python scripts/curation/create_curated_rlds_datasets.py
```

**Output directory:** `data/curated_datasets/`
- `kaist_greedy_K60/` - Greedy solver, K=60 rollouts
- `kaist_greedy_K100/` - Greedy solver, K=100 rollouts
- `kaist_cpsat_K60/` - CP-SAT solver, K=60 rollouts
- ... and more

Each directory contains RLDS-formatted TFRecord files ready for training.

**Understanding the curation approach:**

The QUBO framework optimizes for:
1. **Coverage**: Select rollouts that cover diverse behavior clusters
2. **Diversity**: Avoid selecting similar/redundant rollouts
3. **Budget**: Select exactly K rollouts

See `QUBO_KAIST_OVERVIEW.md` for detailed mathematical formulation and parameter tuning guidance.

---

## Quick Reference Commands

```bash
# Setup (one-time)
brew services start mongodb-community@7.0
pip install -r requirements.txt -e .

# Download dataset (replace YOUR_DATASET_NAME)
oxe_download --dataset YOUR_DATASET_NAME --path ./data/oxe

# Test ingestion with small sample (RECOMMENDED FIRST!)
python scripts/ingest_oxe_dataset.py \
    --dataset-filename YOUR_DATASET_FILENAME \
    --dataset-formalname "Your Dataset Name" \
    --max-episodes 5 --skip-grounding

# Full ingestion
python scripts/ingest_oxe_dataset.py \
    --dataset-filename YOUR_DATASET_FILENAME \
    --dataset-formalname "Your Dataset Name"

# Visualize and filter
streamlit run src/ares/app/webapp.py
# 1. Select dataset
# 2. Apply filters
# 3. Export CSV with IDs

# QUBO-based curation (Steps 6-8)
python scripts/curation/build_window_qubo_instance.py
python scripts/curation/run_batch_experiments.py
python scripts/curation/create_curated_rlds_datasets.py

# Export to RLDS (if you have a custom export script)
python scripts/YOUR_DATASET/export_to_rlds.py --ids-csv filtered.csv --output-dir ./output

# Stop MongoDB when done (optional)
brew services stop mongodb-community@7.0
```

---
