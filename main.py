
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

parser = ArgumentParser()
parser.add_argument("--model_name_or_path",
                    default="rrivera1849/LUAR-CRUD",
                    help="Model name (HF ID), or local path to the model.")
parser.add_argument("--dataset",
                    default="20_Newsgroups_Fixed",
                    choices=DATASET_REGISTRY,
                    help="Dataset to evaluate on.")
parser.add_argument("--cache_dir",
                    default="/tmp/riverasoto1",
                    help="Location to a temporary cache directory.")
parser.add_argument("-e", "--episode_sizes", type=int, action="append", default=None, required=True,
                    help="Number of atomic units to form writing sample.")
parser.add_argument("--n_episodes_per_class", type=int, default=50,
                    help="Number of examples per class.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for embedding.")
parser.add_argument("--force_reload", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=43)
FLAGS = parser.parse_args()
set_seed(FLAGS.seed)

def get_scores_path(episode_size, task_name):
    model_str = os.path.basename(FLAGS.model_name_or_path)
    if model_str == "":
        model_str = os.path.basename(os.path.dirname(FLAGS.model_name_or_path))
    
    dset_str = os.path.basename(FLAGS.dataset)
    scores_path = f"./outputs/{dset_str}/{model_str}/{episode_size}_{FLAGS.n_episodes_per_class}/{task_name}"
    return scores_path

def main():
    if torch.cuda.is_available():
        print(colored("GPU information", "green"))
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
    else:
        print(colored("WARNING: No GPU detected", "red"))

    memory = Memory(FLAGS.cache_dir, verbose=1)
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

    for episode_size in FLAGS.episode_sizes:
    
        dset_loader = DatasetLoader(
            dataset_name=FLAGS.dataset,
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

        for task_name, task_config in config.get("tasks", {}).items():
            # Dynamically import and instantiate processor
            processor_module = importlib.import_module(f"processors.{task_config['processor']}")
            processor_class = getattr(processor_module, f"{task_config['processor'].replace('_', ' ').title().replace(' ', '')}Processor")
            processor = processor_class()

            # Process data
            processed_data = processor.process(X, y)

            # Dynamically import and instantiate task
            task_module = importlib.import_module(f"tasks.{task_name}")
            task_class = getattr(task_module, f"{task_name.replace('_', ' ').title().replace(' ', '')}Task")
            task = task_class()

            # Evaluate task
            metrics = task.evaluate(*processed_data)

            scores_path = get_scores_path(episode_size, task_name)
            os.makedirs(scores_path, exist_ok=True)
            with open(os.path.join(scores_path, "metrics.json"), "w+") as ouf:
                ouf.write(
                    json.dumps(metrics)
                )
            print(f"{FLAGS.dataset}; episode size: {episode_size}; N: {FLAGS.n_episodes_per_class}; Task: {task_name}; Metrics: {metrics}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
