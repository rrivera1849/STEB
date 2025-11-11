
import json
import os
import sys
from argparse import ArgumentParser
import importlib

import numpy as np
import torch
from joblib import Memory
from termcolor import colored
from transformers import set_seed

from dataset_loader import DatasetLoader
from models import MODEL_REGISTRY
from steb_datasets import DATASET_REGISTRY
from utils import CACHE_DIR

def get_supported_datasets(task_name):
    supported_datasets = []
    for dataset_name in DATASET_REGISTRY:
        config_path = os.path.join("steb_datasets", dataset_name, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        if task_name in config.get("tasks", {}):
            supported_datasets.append(dataset_name)
    return supported_datasets

def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True)

    # Base parser for common arguments
    base_parser = ArgumentParser(add_help=False)
    base_parser.add_argument("--model_name_or_path",
                        default="rrivera1849/LUAR-CRUD",
                        help="Model name (HF ID), or local path to the model.")
    base_parser.add_argument("-e", "--episode_sizes", type=int, action="append", default=None, required=True,
                        help="Number of atomic units to form writing sample.")
    base_parser.add_argument("--n_episodes_per_class", type=int, default=50,
                        help="Number of examples per class.")
    base_parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding.")
    base_parser.add_argument("--force_reload", default=False, action="store_true")
    base_parser.add_argument("--seed", type=int, default=43)

    # Clustering task parser
    clustering_parser = subparsers.add_parser("clustering", help="Run clustering task.", parents=[base_parser])
    clustering_parser.add_argument("--dataset",
                                 choices=get_supported_datasets("clustering"),
                                 default=None,
                                 help="Dataset to evaluate on. If not specified, runs on all supported datasets.")

    # Pair classification task parser
    pair_classification_parser = subparsers.add_parser("pair_classification", help="Run pair classification task.", parents=[base_parser])
    pair_classification_parser.add_argument("--dataset",
                                           choices=get_supported_datasets("pair_classification"),
                                           default=None,
                                           help="Dataset to evaluate on. If not specified, runs on all supported datasets.")

    FLAGS = parser.parse_args()
    set_seed(FLAGS.seed)

    def get_scores_path(episode_size, task_name, dataset_name):
        model_str = os.path.basename(FLAGS.model_name_or_path)
        if model_str == "":
            model_str = os.path.basename(os.path.dirname(FLAGS.model_name_or_path))

        dset_str = os.path.basename(dataset_name)
        scores_path = f"./outputs/{dset_str}/{model_str}/{episode_size}_{FLAGS.n_episodes_per_class}/{task_name}"
        return scores_path

    if torch.cuda.is_available():
        print(colored("GPU information", "green"))
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
    else:
        print(colored("WARNING: No GPU detected", "red"))

    memory = Memory(CACHE_DIR, verbose=1)
    memory.clear()

    print(FLAGS.model_name_or_path)
    model_class = None
    for model_cls in MODEL_REGISTRY.values():
        if FLAGS.model_name_or_path in model_cls.supported_models:
            model_class = model_cls
            break
    if model_class is None:
        model_class = MODEL_REGISTRY["hf"]
    model = model_class(FLAGS.model_name_or_path)

    @memory.cache
    def extract_features(dataset, episode_size, n_episodes_per_class, batch_size):
        episodes_by_label = {}
        for label, episodes in dataset.items():
            episodes_by_label[label] = [episodes[i:i+episode_size] for i in range(0, len(episodes), episode_size)]
            assert len(episodes_by_label[label]) == n_episodes_per_class
            assert all([len(episode) == episode_size for episode in episodes_by_label[label]])
        
        all_episodes = [episode for label, episodes in episodes_by_label.items() for episode in episodes]
        y = [label for label, episodes in episodes_by_label.items() for _ in episodes]

        X = model.embed_multiple(all_episodes, batch_size)
        return X, y

    datasets_to_run = [FLAGS.dataset] if FLAGS.dataset else get_supported_datasets(FLAGS.task)

    for dataset_name in datasets_to_run:
        for episode_size in FLAGS.episode_sizes:
        
            dset_loader = DatasetLoader(
                dataset_name=dataset_name,
                episode_size=episode_size,
                n_episodes_per_class=FLAGS.n_episodes_per_class,
                force_reload=FLAGS.force_reload,
            )
            dataset = dset_loader.load()

            if len(dataset) <= 0:
                continue

            X, y = extract_features(dataset, episode_size, FLAGS.n_episodes_per_class, FLAGS.batch_size)

            # Load dataset config to get tasks
            with open(dset_loader.config_path) as f:
                config = json.load(f)

            task_name = FLAGS.task
            task_config = config.get("tasks", {}).get(task_name)

            if not task_config:
                print(colored(f"Task '{task_name}' not supported by dataset '{dataset_name}'. Skipping.", "yellow"))
                continue

            # Dynamically import and instantiate processor
            processor_module = importlib.import_module(f"processors.{task_config['processor']}")
            processor_class_name = f"{task_config['processor'].replace('_', ' ').title().replace(' ', '')}Processor"
            processor_class = getattr(processor_module, processor_class_name)
            processor = processor_class()

            # Process data
            processed_data = processor.process(X, y)

            # Dynamically import and instantiate task
            task_module = importlib.import_module(f"tasks.{task_name}")
            task_class_name = f"{task_name.replace('_', ' ').title().replace(' ', '')}Task"
            task_class = getattr(task_module, task_class_name)
            task = task_class()

            # Evaluate task
            metrics = task.evaluate(*processed_data)

            scores_path = get_scores_path(episode_size, task_name, dataset_name)
            os.makedirs(scores_path, exist_ok=True)
            with open(os.path.join(scores_path, "metrics.json"), "w+") as ouf:
                ouf.write(
                    json.dumps(metrics)
                )
            print(f"{dataset_name}; episode size: {episode_size}; N: {FLAGS.n_episodes_per_class}; Task: {task_name}; Metrics: {metrics}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
