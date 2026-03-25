# STEB: Style Text Embedding Benchmark

STEB is a framework for evaluating style text embeddings across a variety of tasks and datasets. It is modular and extensible, making it straightforward to add new models, datasets, and evaluation tasks.

## Installation

```bash
git clone https://github.com/rrivera1849/STEB.git
cd STEB
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Downloading Datasets

Some datasets need to be downloaded before use. The `download_datasets.sh` script handles this:

```bash
./download_datasets.sh
```

The script skips datasets that have already been downloaded. Use `--purge` to force a clean re-download.

## Configuration

By default, STEB looks for raw datasets in `./raw_datasets` relative to the working directory. To run from another directory, set the `STEB_RAW_DATASETS_DIR` environment variable:

```bash
export STEB_RAW_DATASETS_DIR="/path/to/your/raw_datasets"
```

Other configurable paths (via environment variables or `config.ini`):

| Variable | Default | Description |
|---|---|---|
| `STEB_RESULTS_DIR` | `./results` | Where evaluation results are saved |
| `STEB_CACHE_DIR` | `~/.cache/steb` | Embedding cache directory |
| `STEB_PROCESSED_DATA_DIR` | `~/.local/share/steb/processed_datasets` | Processed dataset cache |
| `STEB_RAW_DATASETS_DIR` | `./raw_datasets` | Raw downloaded datasets |

These can also be set in a `config.ini` file (in the current directory or `~/.steb/config.ini`):

```ini
[Application_Paths]
cache_dir = /path/to/your/cache
processed_dataset_dir = /path/to/your/processed_datasets
results_dir = /path/to/your/results
raw_datasets_dir = /path/to/your/raw_datasets
```

## Usage

### Programmatic

```python
import steb

model = steb.get_model("rrivera1849/LUAR-MUD")
datasets = steb.get_supported_datasets(task_name="clustering")
steb.evaluate(model, datasets=datasets, task_name="clustering", episode_sizes=[1])
```

### CLI

```bash
# List datasets for a task
steb clustering --list-datasets

# Run all tasks on all datasets
steb all "rrivera1849/LUAR-MUD" -e 1

# Run a specific task on a specific dataset
steb clustering "rrivera1849/LUAR-MUD" --dataset "sms_spam" -e 1

# Run with a preset configuration
steb --preset fast "rrivera1849/LUAR-MUD"
```

### Utility Commands

```bash
# Validate all dataset config.json files
steb validate

# Scaffold a new dataset
steb new-dataset my_dataset --type huggingface
steb new-dataset my_dataset --type custom
```

## Tasks

### Clustering

Evaluates how well embeddings form clusters that align with style-based class labels. Episodes are embedded and K-Means clustering is applied. Quality is measured using [V-measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html).

```bash
steb clustering rrivera1849/LUAR-MUD --dataset corpus-of-diverse-styles -e 5
```

### All-to-All Pair Classification

Evaluates whether embeddings can distinguish same-class vs. different-class text groups using cosine similarity across all pairs.

**Metrics:** EER (lower is better), AUC (higher is better), AUC@FPR at thresholds 0.01, 0.05, 0.10, 0.20, 0.30, 0.50.

```bash
steb all_to_all_pair_classification rrivera1849/LUAR-MUD --dataset corpus-of-diverse-styles -e 5
```

### Pre-defined Pair Classification

Same as all-to-all, but operates on datasets with pre-defined pairs (e.g., authorship verification). Episode size and episodes-per-class are set automatically.

```bash
steb pre_defined_pair_classification rrivera1849/LUAR-MUD --dataset pan15_authorship_verification_english_test
```

### Order Alignment

Evaluates how well embeddings preserve the ordering of graded stylistic dimensions (e.g., formality levels). Given two text sets ordered by style intensity, the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) finds the optimal alignment between positions. This generalizes the STEL task.

The task includes a **distractor variant** where items from one set are injected into another, testing robustness to style distractors. See the [Hungarian algorithm documentation](documentation/hungarian-algorithm.md) for details.

**Metrics:** `acc_mean` (baseline alignment accuracy), `distractor_acc_mean` (accuracy with distractors).

### Retrieval

Evaluates how well embeddings retrieve style-matched texts. Given query and target sets, measures how well the correct target is ranked.

**Metrics:** MRR, Mean Rank, Recall@K (K = 1, 8, 16, 32, 64, 128).

```bash
steb retrieval rrivera1849/LUAR-MUD --dataset <dataset_name> -e 50 --n-episodes-per-class 1
```

For datasets with the standard JSONL format (`text`, `label`, `is_query` fields), use the default retrieval loader in `steb/loaders/retrieval.py`. See `steb/steb_datasets/dummy_retrieval/config.json` for an example.

### Probing

Trains a logistic regression probe on frozen embeddings to evaluate what linguistic properties are encoded. Uses train/val/test splits defined per-sample in the dataset.

**Metrics:** Per-task accuracy and average accuracy across all probing tasks.

## Developer Guide

### Supported Models

STEB supports three model types:

- **Encoder models** (`HFModel`): Bidirectional transformers (BERT, RoBERTa, etc.) using mean pooling.
- **Causal models** (`CausalModel`): Auto-regressive LMs (GPT-2, Llama, Mistral, etc.) using last-token pooling.
- **LUAR models** (`LUARModel`): Dedicated support for LUAR-CRUD and LUAR-MUD.

Model type is auto-detected from the HuggingFace config. Encoder vs. causal routing happens automatically in `get_model()`.

### Adding a New Model

1. Create a new file in `steb/models/` (e.g., `steb/models/my_model.py`).
2. Inherit from `STEBModel` and implement `embed_multiple`.
3. Register in `steb/models/__init__.py` by adding to `MODEL_REGISTRY`.

### Adding a New Dataset

The fastest way to add a dataset:

```bash
# Scaffold the directory and config
steb new-dataset my_dataset --type huggingface

# Edit the generated config.json (set path, split, tasks, etc.)

# Validate your config
steb validate
```

This creates `steb/steb_datasets/my_dataset/config.json` (and a stub `loader.py` for custom datasets). The dataset is automatically discovered once the config exists.

#### Config Schema

```json
{
  "dataset_name": "my_dataset",
  "type": "huggingface",
  "record_handler": {
    "text_getter": "text",
    "label_getter": "label"
  },
  "loader_kwargs": {
    "path": "huggingface/dataset-id",
    "split": "train"
  },
  "tasks": {
    "clustering": { "processor": "clustering" },
    "all_to_all_pair_classification": { "processor": "all_to_all_pair_classification" }
  }
}
```

For custom datasets, replace `loader_kwargs` with `data_dir` and `loader_function`. See existing configs in `steb/steb_datasets/` for examples.

**Loader location convention:**
- Shared loaders (used by multiple datasets): `steb/loaders/`
- Dataset-specific loaders: `steb/steb_datasets/<name>/loader.py`

### Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The test suite includes unit tests and integration tests that run every task type end-to-end on dummy datasets.
