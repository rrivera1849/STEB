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
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

## Downloading Datasets

Some of the datasets used in this benchmark need to be downloaded manually. The `download_datasets.sh` script will download and set up these datasets for you.

```bash
./download_datasets.sh
```

**Note:** The `jigsaw_toxicity_pred` dataset is downloaded via `gdown`, which may require you to have `gdown` installed and authenticated with your Google account.

## Running Evaluations

You can run evaluations for different tasks using the `main.py` script. The two main tasks are `clustering` and `pair_classification`.

### Clustering

Here's an example of how to run a clustering evaluation on the `20_Newsgroups_Fixed` dataset with the `LUAR-MUD` model:

```bash
python main.py clustering \
    --dataset 20_Newsgroups_Fixed \
    --model_name_or_path "rrivera1849/LUAR-MUD" \
    -e 5
```

### Pair Classification

Here's an example of how to run a pair classification evaluation on the `enron_authorship_corpus` dataset with the `LUAR-MUD` model:

```bash
python main.py pair_classification \
    --dataset enron_authorship_corpus \
    --model_name_or_path "rrivera1849/LUAR-MUD" \
    -e 1
```

### Running All Tests

You can run a full suite of tests using the `test.sh` script. This will run evaluations for all supported datasets.

```bash
./test.sh
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

1.  Create a new Python file in the `models` directory (e.g., `models/my_model.py`).
2.  In this file, create a class that inherits from `STEBModel` (from `models.base`) and implements the `embed_single` and `embed_multiple` methods.
3.  Register your new model in `models/__init__.py` by adding it to the `MODEL_REGISTRY` dictionary.

### Adding a New Dataset

To add a new dataset, you need to:

1.  Create a new subdirectory in the `steb_datasets` directory with the name of your dataset (e.g., `steb_datasets/my_dataset`).
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
