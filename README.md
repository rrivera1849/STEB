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
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Downloading Datasets

Some of the datasets used in this benchmark need to be downloaded manually. The `download_datasets.sh` script will download and set up these datasets for you.

```bash
./download_datasets.sh
```

**Note:** The `jigsaw_toxicity_pred` dataset is downloaded via `gdown`, which may require you to have `gdown` installed and authenticated with your Google account.

## Running Evaluations

You can run evaluations for different tasks using the `main.py` script. The two main tasks (for now) are `clustering` and `pair_classification`. The outputs will be stored under a new `./outputs` directory.

### Clustering

The clustering task evaluates how well embeddings form clusters that align with style-based class labels (e.g., authorship). 
Each class is represented by multiple "episodes" (groups of texts from the same style-based class). These episodes are embedded (e.g., by averaging embeddings for each text in the group), and K-Means clustering is applied to the embeddings. 
The quality of the clustering is measured using [V-measure score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html), which is the harmonic mean of homogeneity (all cluster members belong to the same class) and completeness (all members of a class are in the same cluster). 

Here's an example of how to run a clustering evaluation on the `corpus-of-diverse-styles` dataset with the `LUAR-MUD` model using episodes of size 5:

```bash
python main.py clustering \
    --dataset corpus-of-diverse-styles \
    --model_name_or_path "rrivera1849/LUAR-MUD" \
    -e 5
```

### Pair Classification

The pair classification task evaluates how well embeddings can distinguish whether two text groups (again controlled via episode parameter) come from the same style-based class (e.g., same author) or different classes. This is calculated using cosine similarity between the embeddings of the two text groups. 

**Metrics:**
- **EER (Equal Error Rate)**: The error rate at the threshold where false positive rate equals false negative rate. Lower is better.
- **AUC (Area Under ROC Curve)**: Measures overall discriminative ability. Higher is better (range 0-1).
- **AUC@FPR**: AUC calculated at specific false positive rate thresholds (0.01, 0.05, 0.10, 0.20, 0.30, 0.50), useful for understanding performance at different operating points.

Here's an example of how to run a pair classification evaluation on the `corpus-of-diverse-styles` dataset with the `LUAR-MUD` model using episodes of size 5:

```bash
python main.py pair_classification \
    --dataset corpus-of-diverse-styles \
    --model_name_or_path "rrivera1849/LUAR-MUD" \
    -e 5
```

### Order Alignment

The order alignment task evaluates how well embeddings can be used to align the stylistic order of one unordered set of texts to that of another, ordered set of texts. The ordered set of texts has to be meaningfully ordered in a style-based graded dimension (e.g., the first text is the least formal, the second text is more formal, and so on). The other set of texts has to vary along the same graded dimension, but is unordered. There is no training involved in this task, it evalautes the intrinsic sensitivity of the embeddings to the investigated graded stylistic dimension. This is a generalization of the STEL task. This set of problems includes tasks where the unordered set includes "distractor" texts that are too dissimilar in style to be aligned to any text in the ordered set, for example, they have a completely different style, but might be interesting to test because they are about the same topic. In a setup with a one element ordered set, and with a two element unordered set including one distractor, this is equivalent to the STEL-or-Content task.

Method: The unordered and ordered sets of texts are embedded and then aligned by reframing the problem as an Assignment problem and using scipy's optimize.linear_sum_assignment, which maximizes the total cosine similarity between the embeddings of the ordered set and the selected embeddings at the same position of the newly ordered set. 

Evaluation: The quality of the alignment is measured using Spearman's rank correlation coefficient between the predicted order and the true order.

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