# Python API

STEB can be used programmatically for more control over evaluations.

## Basic Usage

```python
import steb

# Load a model (type is auto-detected)
model = steb.get_model("rrivera1849/LUAR-MUD")

# Get datasets that support a task
datasets = steb.get_supported_datasets(task_name="clustering")

# Run evaluation
steb.evaluate(
    model,
    datasets=datasets,
    task_name="clustering",
    episode_sizes=[1],
)
```

## Loading Models

`get_model` auto-detects the model type from the HuggingFace config:

```python
# Encoder model (BERT, RoBERTa, etc.)
model = steb.get_model("roberta-base")

# Causal model (GPT-2, Llama, etc.)
model = steb.get_model("gpt2")

# LUAR model
model = steb.get_model("rrivera1849/LUAR-MUD")
```

## Querying Datasets

```python
# All registered datasets
all_datasets = steb.get_all_datasets()

# Datasets supporting a specific task
clustering_datasets = steb.get_supported_datasets(task_name="clustering")
```

## Running Evaluations

The `evaluate` function runs a model on one or more datasets:

```python
steb.evaluate(
    model,
    datasets=["corpus-of-diverse-styles"],
    episode_sizes=[1, 5],
    task_name="clustering",         # None to run all tasks
    n_episodes_per_class=50,
    batch_size=32,
    force_reload=False,
    progress_bar=True,
    output_folder="./results",
    seed=42,
)
```

Results are saved as JSON to `{output_folder}/{dataset}/{model}/{episode_size}_{n_episodes_per_class}/{task}/metrics.json`.
