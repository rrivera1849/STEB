
import json
import os
import importlib
from typing import List, Dict, Optional

from joblib import Memory
from termcolor import colored
from tqdm import tqdm
from transformers import set_seed

from .dataset_loader import DatasetLoader
from .models import MODEL_REGISTRY
from .steb_datasets import DATASET_REGISTRY
from .utils import CACHE_DIR, RESULTS_DIR


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


def get_all_datasets() -> List[str]:
    """
    Retrieves a list of all available STEB datasets.

    Returns:
        A list of all available dataset names.
    """
    return DATASET_REGISTRY


def get_supported_datasets(task_name: str) -> List[str]:
    """
    Retrieves a list of datasets that support the given task.

    Args:
        task_name: The name of the task.

    Returns:
        A list of supported dataset names.
    """
    supported_datasets = []
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for dataset_name in DATASET_REGISTRY:
        config_path = os.path.join(package_dir, "steb_datasets", dataset_name, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        if task_name in config.get("tasks", {}):
            supported_datasets.append(dataset_name)
    return supported_datasets


def evaluate(
    model,
    datasets: List[str],
    episode_sizes: List[int],
    task_name: Optional[str] = None,
    n_episodes_per_class: int = 50,
    batch_size: int = 32,
    force_reload: bool = False,
    progress_bar: bool = False,
    output_folder: str = RESULTS_DIR,
    seed: int = 42,
):
    """
    Evaluates a model on a list of datasets for a given task.

    Args:
        model: The model to evaluate.
        datasets: A list of dataset names to evaluate on.
        episode_sizes: A list of episode sizes to evaluate.
        task_name: The name of the task to evaluate. If None, runs all tasks.
        n_episodes_per_class: The number of episodes per class.
        batch_size: The batch size for embedding.
        force_reload: Whether to force reload the datasets.
        progress_bar: Whether to show a progress bar.
        output_folder: The folder to save the results to.
        seed: The random seed to use.
    """
    set_seed(seed)
    memory = Memory(CACHE_DIR, verbose=1)
    memory.clear()

    @memory.cache(ignore=['show_progress'])
    def extract_features(dataset, episode_size, n_episodes_per_class, batch_size, show_progress=False):
        """
        Extracts features from the dataset using the specified model.

        Expects dataset format:
            {"label": [[seq1_most, ..., seq1_least], [seq2_most, ..., seq2_least], ...]}

        Each label maps to a list of ordered sequences. Sequences are grouped into
        episodes, then organized by position (most X, ..., least X).

        This function is cached to avoid re-extracting features on subsequent runs.
        """
        episodes_by_label = {}
        for label, text_list in dataset.items():
            # Validate nested list format
            assert text_list and isinstance(text_list[0], list), \
                f"Dataset for label '{label}' must be a list of lists (ordered sequences)"

            seq_len = len(text_list[0])

            seq_len = len(text_list[0])

            if episode_size == -1:
                # Group all sequences into a single episode
                episodes_by_label[label] = [
                    [[seq[pos] for seq in text_list] for pos in range(seq_len)]
                ]
            else:
                # Group sequences into episodes, organize by position
                episodes_by_label[label] = [
                    [[seq[pos] for seq in text_list[i:i+episode_size]] for pos in range(seq_len)]
                    for i in range(0, len(text_list), episode_size)
                ]

            if episode_size != -1:
                assert len(episodes_by_label[label]) == n_episodes_per_class
                assert all([len(episode[0]) == episode_size for episode in episodes_by_label[label]])

        all_episodes = [episode for label, episodes in episodes_by_label.items() for episode in episodes]
        y = [label for label, episodes in episodes_by_label.items() for _ in episodes]

        # assert all elements in all_episodes have same length
        num_positions = len(all_episodes[0])
        assert all([len(episode) == num_positions for episode in all_episodes]), \
            ("All entries must have the same number of positions, "
             "functionality for variable-length text sets not implemented.")
        # Flatten episodes for embedding: [[[pos0s], [pos1s], ...], ...] -> [[pos0s], [pos1s], [pos0s], [pos1s], ...]
        flat_episodes = [position for episode in all_episodes for position in episode]
        # Embed all positions,
        # TODO:
        #   - check if we want this to do sth different depending on the task (if only 0th entry needed, this might do too much)
        #   - check if we want to rewrite embed_multiple to accept the format we actually use
        
        # If episode_size is -1, we force batch_size to 1 because episodes will have different sizes
        current_batch_size = batch_size if episode_size != -1 else 1
        X_flat = model.embed_multiple(flat_episodes, current_batch_size, show_progress=show_progress)
        # Reshape back to episode structure
        X = [X_flat[i:i+num_positions] for i in range(0, len(X_flat), num_positions)]

        return X, y

    dataset_iterator = tqdm(datasets, desc="Evaluating Datasets", disable=not progress_bar)
    for dataset_name in dataset_iterator:
        for episode_size in episode_sizes:
            if episode_size == -1 and task_name != "retrieval":
                raise ValueError("Episode size -1 is only supported for the retrieval task.")

            print(colored(f"--- Evaluating {dataset_name} (episode size: {episode_size}) ---", "cyan"))
            dset_loader = DatasetLoader(
                dataset_name=dataset_name,
                episode_size=episode_size,
                n_episodes_per_class=n_episodes_per_class,
                force_reload=force_reload,
            )
            dataset = dset_loader.load()

            if len(dataset) <= 0:
                continue

            X, y = extract_features(dataset, episode_size, n_episodes_per_class, batch_size, show_progress=progress_bar)

            with open(dset_loader.config_path) as f:
                config = json.load(f)

            tasks_to_run = [task_name] if task_name else config.get("tasks", {}).keys()

            for current_task_name in tasks_to_run:
                print(colored(f"  - Running task: {current_task_name}", "blue"))
                task_config = config.get("tasks", {}).get(current_task_name)
                if not task_config:
                    print(colored(f"Task '{current_task_name}' not supported by dataset '{dataset_name}'. Skipping.", "yellow"))
                    continue

                processor_module = importlib.import_module(f"steb.processors.{task_config['processor']}")
                processor_class_name = f"{task_config['processor'].replace('_', ' ').title().replace(' ', '')}Processor"
                processor_class = getattr(processor_module, processor_class_name)
                processor = processor_class()

                processed_data = processor.process(X, y)

                task_module = importlib.import_module(f"steb.tasks.{current_task_name}")
                task_class_name = f"{current_task_name.replace('_', ' ').title().replace(' ', '')}Task"
                task_class = getattr(task_module, task_class_name)
                task = task_class()

                metrics = task.evaluate(*processed_data)

                model_str = os.path.basename(model.model_name_or_path)
                if model_str == "":
                    model_str = os.path.basename(os.path.dirname(model.model_name_or_path))

                dset_str = os.path.basename(dataset_name)
                scores_path = os.path.join(output_folder, dset_str, model_str, f"{episode_size}_{n_episodes_per_class}", current_task_name)

                os.makedirs(scores_path, exist_ok=True)
                with open(os.path.join(scores_path, "metrics.json"), "w+") as ouf:
                    ouf.write(json.dumps(metrics))

                print(colored(f"    -> Metrics: {metrics}", "green"))
