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

You can also run evaluations from the command line using the `steb` tool. The outputs will be stored under a new `./results` directory by default.

### Listing Datasets

To see the available datasets for a specific task, use the `--list-datasets` flag:

```bash
steb clustering --list-datasets
```

### Running Evaluations

Here are some examples of how to run evaluations:

**Run all tasks on all supported datasets for a given model:**

```bash
steb all "rrivera1849/LUAR-MUD" -e 1
```

**Run the clustering task on all supported datasets:**

```bash
steb clustering "rrivera1849/LUAR-MUD" -e 1
```

**Run the clustering task on a specific dataset:**

```bash
steb clustering "rrivera1849/LUAR-MUD" --dataset "sms_spam" -e 1
```

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
    *   If the `type` is `"huggingface"`, you must include `loader_kwargs`: A dictionary of arguments that will be passed to the `load_dataset` function from the Hugging Face `datasets` library.
    *   If the `type` is `"custom"`, you must include `data_dir`: The path to the dataset's data directory. You will also need to create a `loader.py` file in the same directory and specify the loader function in the `config.json` with the `loader_function` key.
    *   If your dataset requires a custom label transformation, you can add the function to the `loader.py` file and specify it in the `config.json` with the `label_getter_function` key.

Your new dataset will be automatically discovered and made available as a choice for the `--dataset` argument.

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