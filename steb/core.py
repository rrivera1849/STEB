
import json
import os
import importlib
from typing import List, Dict, Optional

from joblib import Memory
from termcolor import colored
from transformers import set_seed

from .dataset_loader import DatasetLoader
from .models import MODEL_REGISTRY
from .steb_datasets import DATASET_REGISTRY
from .utils import CACHE_DIR


def get_model(model_name_or_path: str):
    """
    Loads a STEB model.

    Args:
        model_name_or_path: The name or path of the model to load.

    Returns:
        An instance of a STEBModel.
    """
    model_class = None
    for model_cls in MODEL_REGISTRY.values():
        if model_name_or_path in model_cls.supported_models:
            model_class = model_cls
            break
    if model_class is None:
        model_class = MODEL_REGISTRY["hf"]
    return model_class(model_name_or_path)


def get_datasets(datasets: Optional[List[str]] = None) -> List[str]:
    """
    Retrieves a list of available STEB datasets.

    Args:
        datasets: An optional list of dataset names to filter by.

    Returns:
        A list of available dataset names.
    """
    if datasets is None:
        return DATASET_REGISTRY

    # Return intersection of datasets and DATASET_REGISTRY
    return [d for d in datasets if d in DATASET_REGISTRY]


def evaluate(
    model,
    datasets: List[str],
    episode_sizes: List[int],
    n_episodes_per_class: int = 50,
    batch_size: int = 32,
    force_reload: bool = False,
    output_folder: str = "results",
    seed: int = 42,
):
    """
    Evaluates a model on a list of datasets.

    Args:
        model: The model to evaluate.
        datasets: A list of dataset names to evaluate on.
        episode_sizes: A list of episode sizes to evaluate.
        n_episodes_per_class: The number of episodes per class.
        batch_size: The batch size for embedding.
        force_reload: Whether to force reload the datasets.
        output_folder: The folder to save the results to.
        seed: The random seed to use.
    """
    set_seed(seed)
    memory = Memory(CACHE_DIR, verbose=1)
    memory.clear()

    @memory.cache
    def extract_features(dataset, episode_size, n_episodes_per_class, batch_size):
        """
        Extracts features from the dataset using the specified model.
        This function is cached to avoid re-extracting features on subsequent runs.
        """
        episodes_by_label = {}
        for label, episodes in dataset.items():
            episodes_by_label[label] = [episodes[i:i+episode_size] for i in range(0, len(episodes), episode_size)]
            assert len(episodes_by_label[label]) == n_episodes_per_class
            assert all([len(episode) == episode_size for episode in episodes_by_label[label]])

        all_episodes = [episode for label, episodes in episodes_by_label.items() for episode in episodes]
        y = [label for label, episodes in episodes_by_label.items() for _ in episodes]

        X = model.embed_multiple(all_episodes, batch_size)
        return X, y

    for dataset_name in datasets:
        for episode_size in episode_sizes:
            dset_loader = DatasetLoader(
                dataset_name=dataset_name,
                episode_size=episode_size,
                n_episodes_per_class=n_episodes_per_class,
                force_reload=force_reload,
            )
            dataset = dset_loader.load()

            if len(dataset) <= 0:
                continue

            X, y = extract_features(dataset, episode_size, n_episodes_per_class, batch_size)

            with open(dset_loader.config_path) as f:
                config = json.load(f)

            for task_name, task_config in config.get("tasks", {}).items():
                processor_module = importlib.import_module(f"steb.processors.{task_config['processor']}")
                processor_class_name = f"{task_config['processor'].replace('_', ' ').title().replace(' ', '')}Processor"
                processor_class = getattr(processor_module, processor_class_name)
                processor = processor_class()

                processed_data = processor.process(X, y)

                task_module = importlib.import_module(f"steb.tasks.{task_name}")
                task_class_name = f"{task_name.replace('_', ' ').title().replace(' ', '')}Task"
                task_class = getattr(task_module, task_class_name)
                task = task_class()

                metrics = task.evaluate(*processed_data)

                model_str = os.path.basename(model.model_name_or_path)
                if model_str == "":
                    model_str = os.path.basename(os.path.dirname(model.model_name_or_path))

                dset_str = os.path.basename(dataset_name)
                scores_path = f"./{output_folder}/{dset_str}/{model_str}/{episode_size}_{n_episodes_per_class}/{task_name}"

                os.makedirs(scores_path, exist_ok=True)
                with open(os.path.join(scores_path, "metrics.json"), "w+") as ouf:
                    ouf.write(json.dumps(metrics))

                print(f"{dataset_name}; episode size: {episode_size}; N: {n_episodes_per_class}; Task: {task_name}; Metrics: {metrics}")
