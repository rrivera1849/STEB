# STEB: Style Text Embedding Benchmark

STEB (Style Text Embedding Benchmark) is a framework for evaluating style text embeddings across a variety of tasks and datasets. It is designed to be modular and extensible, allowing researchers and developers to easily add new models, datasets, and evaluation tasks.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rrivera1849/STEB.git
    cd STEB
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    ```

## Downloading Datasets

Some of the datasets used in this benchmark need to be downloaded manually. The `download_datasets.sh` script will download and set up these datasets for you.

```bash
./download_datasets.sh
```

**Note:** The `jigsaw_toxicity_pred` dataset is downloaded via `gdown`, which may require you to have `gdown` installed and authenticated with your Google account.

## Programmatic Usage

You can use STEB programmatically to evaluate your models. Here's an example:

```python
import steb

# Select model
model_name = "rrivera1849/LUAR-MUD"
model = steb.get_model(model_name)

# Select datasets for a specific task
datasets = steb.get_supported_datasets(task_name="clustering")

# Evaluate
results = steb.evaluate(model, datasets=datasets, task_name="clustering", episode_sizes=[1])
```

## Running Evaluations from the CLI

You can also run evaluations from the command line using the `steb` tool. The outputs will be stored under a new `./outputs` directory by default, but can be modified with the `--output-folder` flag.

CLI Examples:

```bash
# List available datasets that support the "clustering" task:
steb clustering --list-datasets

# Run all tasks on all supported datasets for a given model:
steb all "rrivera1849/LUAR-MUD" -e 1

# Run the clustering task on all supported datasets:
steb clustering rrivera1849/LUAR-MUD -e 1

# Run the clustering task on a specific dataset:
steb clustering "rrivera1849/LUAR-MUD" --dataset "sms_spam" -e 1
```

## Task Descriptions

In what follows, we detail the various tasks and the metrics they calculate.

### Clustering

The clustering task evaluates how well embeddings form clusters that align with style-based class labels (e.g., authorship). 
Each class is represented by multiple "episodes" (groups of texts from the same style-based class). These episodes are embedded (e.g., by averaging embeddings for each text in the group), and K-Means clustering is applied to the embeddings. 
The quality of the clustering is measured using [V-measure score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html), which is the harmonic mean of homogeneity (all cluster members belong to the same class) and completeness (all members of a class are in the same cluster). 

Here's an example of how to run a clustering evaluation on the `corpus-of-diverse-styles` dataset with the `LUAR-MUD` model using episodes of size 5:

```bash
steb clustering rrivera1849/LUAR-MUD --dataset corpus-of-diverse-styles -e 5
```

### Pair Classification

The pair classification task evaluates how well embeddings can distinguish whether two text groups (again controlled via episode parameter) come from the same style-based class (e.g., same author) or different classes. This is calculated using cosine similarity between the embeddings of the two text groups. 

**Metrics:**
- **EER (Equal Error Rate)**: The error rate at the threshold where false positive rate equals false negative rate. Lower is better.
- **AUC (Area Under ROC Curve)**: Measures overall discriminative ability. Higher is better (range 0-1).
- **AUC@FPR**: AUC calculated at specific false positive rate thresholds (0.01, 0.05, 0.10, 0.20, 0.30, 0.50), useful for understanding performance at different operating points.

Here's an example of how to run a pair classification evaluation on the `corpus-of-diverse-styles` dataset with the `LUAR-MUD` model using episodes of size 5:

```bash
steb pair_classification rrivera1849/LUAR-MUD --dataset corpus-of-diverse-styles -e 5
```

### Order Alignment

The order alignment task evaluates how well embeddings can be used to align the stylistic order of one unordered set of text groups to that of another, ordered set of text groups. Each text group is an episode (controlled via the episode parameter, similar to clustering and pair classification tasks). The ordered set of text groups has to be meaningfully ordered in a style-based graded dimension (e.g., the first text group is the least formal, the second text group is more formal, and so on). The other set of text groups has to vary along the same graded dimension, but is unordered. There is no training involved in this task, it evaluates the intrinsic sensitivity of the embeddings to the investigated graded stylistic dimension. This is a generalization of the STEL task. Every order alignment task also performs a "distractor" version of the same task, where the unordered set includes "distractor" text groups that are too dissimilar in style to be aligned to any text group in the ordered set, for example, they have a completely different style, but might be interesting to test because they are about the same topic. In a setup with a one element ordered set, and with a two element unordered set including one distractor, this is equivalent to the STEL-or-Content task.

**Method**:
- For each pair of text sets with matching labels, embeddings are computed for all positions (most style → least style)
- The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the optimal one-to-one matching between positions that maximizes total cosine similarity
- Negative similarities are clamped to 0, and the cost matrix uses distance (1 - similarity)

**Evaluation**: The task includes two variants:
1. **Baseline**: Compares full position sequences and measures alignment accuracy (proportion of positions correctly matched)
2. **Distractor variant**: Moves the least-intense position and the most-intense position from one sample into another, testing robustness to style distractors

**Metrics:**
- **acc_mean**: Average alignment accuracy across all pairs (baseline variant)
- **distractor_acc_mean**: Average alignment accuracy with style distractors present


## Developer Guide

This guide is for developers who want to extend the STEB framework by adding new models, datasets, or tasks.

### Core Abstractions

The STEB framework is built around three core abstractions:

*   **`STEBModel`**: An abstract base class for style text embedding models. It defines the interface for embedding single texts and episodes (lists of texts).
*   **`Processor`**: An abstract base class for data processors. It defines the interface for processing embeddings and labels before they are passed to a task for evaluation.
*   **`Task`**: An abstract base class for evaluation tasks. It defines the interface for evaluating embeddings and labels and returning a dictionary of metrics.

### Adding a New Model

To add a new model, you need to:

1.  Create a new Python file in the `steb/models` directory (e.g., `steb/models/my_model.py`).
2.  In this file, create a class that inherits from `STEBModel` (from `steb.models.base`) and implements the `embed_single` and `embed_multiple` methods.
3.  Register your new model in `steb/models/__init__.py` by adding it to the `MODEL_REGISTRY` dictionary.

### Adding a New Dataset

To add a new dataset, you need to:

1.  Create a new subdirectory in the `steb/steb_datasets` directory with the name of your dataset (e.g., `steb/steb_datasets/my_dataset`).
2.  Inside this new subdirectory, create a `config.json` file.
3.  This `config.json` file should contain the following keys:
    *   `dataset_name`: The name of the dataset.
    *   `type`: The type of the dataset, either `"huggingface"` or `"custom"`.
    *   `tasks`: A dictionary that maps task names to their configurations.
    *   `record_handler`: Specifies how to extract the text and label from a dataset record. It should have `text_getter` and `label_getter` keys.
        *   For simple cases, `text_getter` and `label_getter` are field names to extract from each record.
        *   For complex processing, you can specify `custom_record_handler_function` in the `record_handler` to point to a custom function in `loader.py` that processes each record and returns the desired format.
    *   If the `type` is `"huggingface"`, you must include `loader_kwargs`: A dictionary of arguments that will be passed to the `load_dataset` function from the Hugging Face `datasets` library.
    *   If the `type` is `"custom"`, you must include `data_dir`: The path to the dataset's data directory. You will also need to create a `loader.py` file in the same directory and specify the loader function in the `config.json` with the `loader_function` key.
    *   If your dataset requires a custom label transformation, you can add the function to the `loader.py` file and specify it in the `config.json` with the `label_getter_function` key.

Your new dataset will be automatically discovered and made available as a choice for the `--dataset` argument.

### Running Tests

To run the test suite locally:

1. Follow the installation steps above (**`pip install -e .`** inside your virtualenv).

2. Install test-only dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. Run the tests from the project root:

   ```bash
   pytest
   ```

Here's an example of such a configuration for a dataset available in HuggingFace:

```
{
  "dataset_name": "billray110/corpus-of-diverse-styles",
  "type": "huggingface",
  "record_handler": {
    "text_getter": "text",
    "label_getter": "label"
  },
  "loader_kwargs": {
    "path": "billray110/corpus-of-diverse-styles",
    "split": "train"
  },
  "tasks": {
    "pair_classification": {
      "processor": "pair_classification"
    },
    "clustering": {
      "processor": "clustering"
    }
  }
}
```