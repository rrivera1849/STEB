
import json
import os
import importlib
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset
from termcolor import colored

from .utils import CACHE_DIR, PROCESSED_DATA_DIR

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

class DatasetLoader(object):
    """
    This class is responsible for loading datasets, processing them into episodes, and caching the results.
    It can handle both Hugging Face datasets and custom local datasets.
    """
    def __init__(
        self,
        dataset_name: str,
        episode_size: int = 5,
        n_episodes_per_class: int = 50,
        force_reload: bool = False,
    ):
        """
        Initializes the DatasetLoader.

        Args:
            dataset_name: The name of the dataset to load.
            episode_size: The number of text samples per episode.
            n_episodes_per_class: The number of episodes to generate for each class.
            force_reload: If True, forces reprocessing of the dataset.
        """
        self.dataset_name = dataset_name
        self.episode_size = episode_size
        self.n_episodes_per_class = n_episodes_per_class
        self.force_reload = force_reload
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

    def load(self):
        """
        Loads, processes, and returns the dataset.
        Handles loading from cache, downloading from Hugging Face or a custom source,
        processing records, and saving the processed data to cache.
        """
        dataset_path = self._get_dataset_path()
        
        if os.path.exists(dataset_path) and not self.force_reload:
            print(colored(f"Loading dataset from {dataset_path}", "green"))
            return json.loads(open(dataset_path, "r").read())

        if self.config["type"] == "huggingface":
            loader_kwargs = self.config["loader_kwargs"]
            loader_kwargs["cache_dir"] = CACHE_DIR
            dataset_iter = load_dataset(**loader_kwargs)
        elif self.config["type"] == "custom":
            loader_module = importlib.import_module(f"steb.steb_datasets.{self.dataset_name}.loader")
            loader_fn = getattr(loader_module, self.config["loader_function"])
            dataset_iter = loader_fn(self.config["data_dir"])
        else:
            raise ValueError(f"Unknown dataset type: {self.config['type']}")

        text_getter = self.config["record_handler"]["text_getter"]
        label_getter = self.config["record_handler"]["label_getter"]
        label_transform = None
        if "label_getter_function" in self.config["record_handler"]:
            loader_module = importlib.import_module(f"steb.steb_datasets.{self.dataset_name}.loader")
            label_transform = getattr(loader_module, self.config["record_handler"]["label_getter_function"])
        custom_record_handler = None
        if "custom_record_handler_function" in self.config["record_handler"]:
            loader_module = importlib.import_module(f"steb.steb_datasets.{self.dataset_name}.loader")
            custom_record_handler = getattr(loader_module, self.config["record_handler"]["custom_record_handler_function"])

        handler = partial(
            record_handler,
            text_getter=text_getter,
            label_getter=label_getter,
            label_transform=label_transform,
            custom_record_handler=custom_record_handler,
        )
        
        N = self.episode_size * self.n_episodes_per_class
        dataset: Dict[str, List[List[str]]] = defaultdict(list)
        valid_labels = self.get_valid_labels(dataset_iter, handler)

        for example in dataset_iter:
            record = handler(example)
            if record is None or record["label"] not in valid_labels:
                continue
            elif len(dataset[record["label"]]) >= N:
                continue
            dataset[record["label"]].append(record["text"])

        dataset = {k: v for k, v in dataset.items() if len(v) == N}
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, "w") as f:
            print(f"Saving dataset to {dataset_path}")
            f.write(json.dumps(dataset))
        return dataset
    
    def _get_dataset_path(self):
        """
        Generates the file path for the cached, processed dataset.
        """
        base_str = f"{os.path.basename(self.dataset_name)}_{self.n_episodes_per_class}_{self.episode_size}"
        return os.path.join(PROCESSED_DATA_DIR, base_str + ".json")
    
    def get_valid_labels(self, dataset_iter, handler):
        """
        Gets all the labels for classes that have enough samples.
        A class is considered valid if it has at least `self.episode_size * self.n_episodes_per_class` samples.
        """
        N = self.episode_size * self.n_episodes_per_class

        label_to_count: Dict[str, int] = defaultdict(int)
        for example in dataset_iter:
            record = handler(example)
            if record is None:
                continue
            label_to_count[record["label"]] += 1

        label_to_count = {k: v for k, v in label_to_count.items() if v >= N}
        valid_labels = list(label_to_count.keys())

        if not valid_labels:
            raise warnings.warn(f"No valid labels found with at least {N} samples in dataset: {self.dataset_name}. "
                          f"This might be expected for dummy datasets.")

        return valid_labels
