import importlib
import json
import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import load_dataset
from termcolor import colored

from .utils import CACHE_DIR, PROCESSED_DATA_DIR, RAW_DATASETS_DIR


def record_handler(
    example: Dict[str, Any],
    text_getter: str,
    label_getter: str,
    label_transform: Optional[Callable[[Any], Any]] = None,
    custom_record_handler: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extracts text and a single label from a dataset record.

    Args:
        example: The input data record.
        text_getter: The key to access the text in the example.
        label_getter: The key to access the label in the example.
        label_transform: An optional function to transform the label.
        custom_record_handler: An optional custom function to preprocess the record,
            returns None to skip the record, or a modified record with text_getter and label_getter keys.

    Returns:
        A dictionary with "text" and "label" keys, or None if the text or label is missing.
    """
    if custom_record_handler:
        example = custom_record_handler(example)
        if example is None:
            return None

    text = example[text_getter]
    label = example[label_getter]
    if isinstance(label, list):
        label = label[0] if label else None

    if text is None or label is None:
        return None

    if label_transform:
        label = label_transform(label)

    if isinstance(text, str):
        text = [text]
    elif isinstance(text, list):
        text = [t for t in text if isinstance(t, str)]
    else:
        return None

    if len(text) == 0:
        return None

    return {"text": text, "label": label}


class DatasetLoader:
    """
    This class is responsible for loading datasets, processing them into episodes, and caching the results.
    It can handle both Hugging Face datasets and custom local datasets.
    """
    AUTO_MIN_EPISODES = 25
    AUTO_MAX_EPISODES = 200

    def __init__(
        self,
        dataset_name: str,
        episode_size: int = 5,
        n_episodes_per_class: Union[int, str] = 50,
        force_reload: bool = False,
        seed: int = 42,
        task_name: Optional[str] = None,
    ):
        """
        Initializes the DatasetLoader.

        Args:
            dataset_name: The name of the dataset to load.
            episode_size: The number of text samples per episode.
            n_episodes_per_class: The number of episodes to generate for each class.
                Use "auto" to adaptively pick the value that preserves all classes,
                clamped to [AUTO_MIN_EPISODES, AUTO_MAX_EPISODES].
            force_reload: If True, forces reprocessing of the dataset.
            seed: The random seed used for sampling (included in cache key).
            task_name: The task to load data for. When a task defines its own
                record_handler in config.json, that handler overrides the
                top-level record_handler. This also affects the cache key so
                that tasks with different handlers get separate cached files.
        """
        self.dataset_name = dataset_name
        self.episode_size = episode_size
        self.n_episodes_per_class = n_episodes_per_class
        self.force_reload = force_reload
        self.seed = seed
        self.task_name = task_name
        self.config_path, self.config = self._load_config()

    def _load_config(self):
        """
        Loads the config.json file for the specified dataset.
        """
        package_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(package_dir, "steb_datasets", self.dataset_name, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found for dataset: {self.dataset_name}")
        with open(config_path, "r") as f:
            return config_path, json.load(f)

    def _load_source_and_handler(self):
        """
        Loads the raw dataset and builds the record handler.

        Returns:
            A tuple of (dataset_iter, handler) where dataset_iter is the raw
            dataset iterable and handler is a partial-applied record_handler.
        """
        loader_module_name = self.config.get("loader_module", f"steb.steb_datasets.{self.dataset_name}.loader")

        if self.config["type"] == "huggingface":
            loader_kwargs = dict(self.config["loader_kwargs"])
            loader_kwargs["cache_dir"] = CACHE_DIR
            dataset_iter = load_dataset(**loader_kwargs)
        elif self.config["type"] == "custom":
            loader_module = importlib.import_module(loader_module_name)
            loader_fn = getattr(loader_module, self.config["loader_function"])
            dataset_iter = loader_fn(os.path.join(RAW_DATASETS_DIR, self.config["data_dir"]))
        else:
            raise ValueError(f"Unknown dataset type: {self.config['type']}")

        effective_rh = self._get_effective_record_handler()
        text_getter = effective_rh.get("text_getter")
        label_getter = effective_rh.get("label_getter")

        label_transform = None
        if "label_getter_function" in effective_rh:
            lm = importlib.import_module(loader_module_name)
            label_transform = getattr(lm, effective_rh["label_getter_function"])

        custom_record_handler = None
        if "custom_record_handler_function" in effective_rh:
            lm = importlib.import_module(loader_module_name)
            custom_record_handler = getattr(lm, effective_rh["custom_record_handler_function"])

        handler = partial(
            record_handler,
            text_getter=text_getter,
            label_getter=label_getter,
            label_transform=label_transform,
            custom_record_handler=custom_record_handler,
        )

        return dataset_iter, handler

    def load(self):
        """
        Loads, processes, and returns the dataset.

        Handles loading from cache, downloading from Hugging Face or a custom source,
        processing records, and saving the processed data to cache.
        When ``n_episodes_per_class`` is ``"auto"``, the value is resolved from
        the dataset before caching so the cache key reflects the actual count.
        """
        # For non-auto mode, try cache first (before loading the dataset)
        if self.n_episodes_per_class != "auto":
            dataset_path = self._get_dataset_path()
            if os.path.exists(dataset_path) and not self.force_reload:
                print(colored(f"Loading dataset from {dataset_path}", "green"))
                with open(dataset_path, "r") as f:
                    return json.loads(f.read())

        dataset_iter, handler = self._load_source_and_handler()
        label_counts = self._count_labels(dataset_iter, handler)

        # Resolve "auto" n_episodes_per_class from the label counts
        if self.n_episodes_per_class == "auto":
            self.n_episodes_per_class = self._resolve_auto_episodes(label_counts)

        # Now that n_episodes_per_class is resolved, check cache
        dataset_path = self._get_dataset_path()
        if os.path.exists(dataset_path) and not self.force_reload:
            print(colored(f"Loading dataset from {dataset_path}", "green"))
            with open(dataset_path, "r") as f:
                return json.loads(f.read())

        if self.episode_size == -1:
            N = 1
        else:
            N = self.episode_size * self.n_episodes_per_class

        valid_labels = self._get_valid_labels_from_counts(label_counts, N)

        dataset: Dict[str, List[List[str]]] = defaultdict(list)

        for example in dataset_iter:
            record = handler(example)
            if record is None or record["label"] not in valid_labels:
                continue
            elif self.episode_size != -1 and len(dataset[record["label"]]) >= N:
                continue
            dataset[record["label"]].append(record["text"])

        #   Unsure if this is necessary at this point, we should've ensured that
        # everything is the same size
        if self.episode_size != -1:
            dataset = {k: v for k, v in dataset.items() if len(v) == N}

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, "w") as f:
            print(f"Saving dataset to {dataset_path}")
            f.write(json.dumps(dataset))

        return dataset

    def _get_effective_record_handler(self) -> Dict[str, Any]:
        """
        Returns the record handler config for the current task.

        If the task defines its own ``record_handler`` in ``config.json``,
        it is merged on top of the top-level ``record_handler`` so that
        task-specific keys (e.g. ``custom_record_handler_function``) override
        the defaults while inheriting anything not explicitly set.

        Returns:
            A merged record handler dictionary.
        """
        base_rh = dict(self.config["record_handler"])
        if self.task_name is None:
            return base_rh

        task_config = self.config.get("tasks", {}).get(self.task_name, {})
        task_rh = task_config.get("record_handler")
        if task_rh is None:
            return base_rh

        merged = {**base_rh, **task_rh}
        return merged

    def _has_task_specific_record_handler(self) -> bool:
        """
        Checks whether the current task overrides the top-level record handler.

        Returns:
            True if the task defines its own record_handler block.
        """
        if self.task_name is None:
            return False
        task_config = self.config.get("tasks", {}).get(self.task_name, {})
        return "record_handler" in task_config

    def _get_dataset_path(self) -> str:
        """
        Generates the file path for the cached, processed dataset.

        The cache key includes the seed to avoid using stale data when the
        seed changes. When the current task has a task-specific record
        handler, the task name is appended to the cache key so that tasks
        with different handlers get separate cached files.

        Returns:
            The file path for the cached dataset.
        """
        base_str = f"{os.path.basename(self.dataset_name)}_{self.n_episodes_per_class}_{self.episode_size}_seed{self.seed}"
        if self._has_task_specific_record_handler():
            base_str += f"_{self.task_name}"
        return os.path.join(PROCESSED_DATA_DIR, base_str + ".json")

    def preview(self) -> Dict[str, Any]:
        """
        Returns dataset statistics without collecting data.

        Loads the dataset source, counts labels, resolves "auto" n_episodes_per_class,
        and reports which classes would be kept or dropped.

        Returns:
            A dict with keys: total_classes, kept_classes, dropped_classes,
            dropped_labels, n_episodes_per_class, episode_size, min_class_count,
            samples_per_class.
        """
        dataset_iter, handler = self._load_source_and_handler()
        label_counts = self._count_labels(dataset_iter, handler)

        if self.n_episodes_per_class == "auto":
            resolved = self._resolve_auto_episodes(label_counts)
        else:
            resolved = self.n_episodes_per_class

        if self.episode_size == -1:
            N = 1
        else:
            N = self.episode_size * resolved

        min_count = min(label_counts.values()) if label_counts else 0
        dropped = {k: v for k, v in label_counts.items() if v < N}

        return {
            "total_classes": len(label_counts),
            "kept_classes": len(label_counts) - len(dropped),
            "dropped_classes": len(dropped),
            "dropped_labels": dropped,
            "n_episodes_per_class": resolved,
            "episode_size": self.episode_size,
            "min_class_count": min_count,
            "samples_per_class": N,
        }

    def _count_labels(
        self,
        dataset_iter,
        handler,
    ) -> Dict[str, int]:
        """
        Counts the number of samples per label in the dataset.

        Args:
            dataset_iter: The dataset iterator.
            handler: The record handler function.

        Returns:
            A dictionary mapping labels to their sample counts.
        """
        label_to_count: Dict[str, int] = defaultdict(int)
        for example in dataset_iter:
            record = handler(example)
            if record is None:
                continue
            label_to_count[record["label"]] += 1
        return label_to_count

    def _resolve_auto_episodes(
        self,
        label_counts: Dict[str, int],
    ) -> int:
        """
        Computes n_episodes_per_class that preserves all classes, clamped to
        [AUTO_MIN_EPISODES, AUTO_MAX_EPISODES].

        Args:
            label_counts: A dictionary mapping labels to their sample counts.

        Returns:
            The resolved n_episodes_per_class value.
        """
        if self.episode_size == -1:
            return 1

        min_samples = min(label_counts.values())
        computed = min_samples // self.episode_size
        resolved = max(self.AUTO_MIN_EPISODES, min(computed, self.AUTO_MAX_EPISODES))

        print(colored(
            f"  Auto n_episodes_per_class for '{self.dataset_name}': "
            f"smallest class has {min_samples} samples, "
            f"episode_size={self.episode_size} -> "
            f"computed={computed}, resolved={resolved}",
            "blue",
        ))

        return resolved

    def _get_valid_labels_from_counts(
        self,
        label_counts: Dict[str, int],
        N: int,
    ) -> List[str]:
        """
        Filters labels that have at least N samples.

        Args:
            label_counts: A dictionary mapping labels to their sample counts.
            N: The minimum number of samples required per class.

        Returns:
            A list of valid label strings.
        """
        total_classes = len(label_counts)
        print(colored(
            f"  Dataset '{self.dataset_name}': {total_classes} classes found, "
            f"need {N} samples per class",
            "blue",
        ))

        dropped = {k: v for k, v in label_counts.items() if v < N}
        if dropped:
            print(colored(f"  Dropping {len(dropped)} class(es) with insufficient samples:", "yellow"))
            for label, count in sorted(dropped.items(), key=lambda x: x[1]):
                print(colored(f"    - '{label}': {count}/{N} samples", "yellow"))

        valid_labels = [k for k, v in label_counts.items() if v >= N]

        if not valid_labels:
            raise ValueError(
                f"No valid labels found with at least {N} samples in dataset: {self.dataset_name}. "
                f"This might be expected for dummy datasets."
            )

        print(colored(f"  Keeping {len(valid_labels)}/{total_classes} classes", "green"))
        return valid_labels
