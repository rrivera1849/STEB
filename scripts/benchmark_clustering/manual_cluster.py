"""Manual cluster tables built from a YAML cluster definition file."""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .config import TASK_METRICS
from .discovery import discover_all_scores


def load_manual_clusters(
    clusters_path: str,
) -> Dict[str, List[str]]:
    """Load manual dataset clusters from a YAML file.

    Args:
        clusters_path: Path to the YAML file with cluster definitions.

    Returns:
        A dict mapping cluster name to list of dataset names.
    """
    with open(clusters_path) as f:
        raw = yaml.safe_load(f)

    clusters = {}
    for name, config in raw.items():
        clusters[name] = config["datasets"]
    return clusters


def build_manual_cluster_tables(
    results_dir: str,
    clusters: Dict[str, List[str]],
    episode_params: Optional[str],
    include_excluded: bool = False,
    complete_datasets: bool = False,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, List[str]]]]:
    """Build one table per manual cluster.

    For each cluster, produces a DataFrame where rows are models and columns
    are tasks. Each cell is the average metric across datasets in that cluster
    that support the task.

    Args:
        results_dir: Path to the root results directory.
        clusters: Mapping from cluster name to list of dataset names.
        episode_params: Episode params filter (e.g. '1_50').
        include_excluded: If True, include semantic and non-English datasets.
        complete_datasets: If True, within each (cluster, task) group, drop
            datasets that not all models have results for.

    Returns:
        A tuple of:
          - A dict mapping cluster name to a DataFrame (models x tasks).
          - A dict mapping cluster name to a dict of task column name to
            list of dataset names included in that column.
    """
    all_scores = discover_all_scores(results_dir, include_excluded)
    if not all_scores:
        return {}, {}

    # Invert clusters: dataset -> cluster name
    dataset_to_cluster: Dict[str, str] = {}
    for cluster_name, datasets in clusters.items():
        for ds in datasets:
            dataset_to_cluster[ds] = cluster_name

    # Collect per-dataset scores: (cluster, task, dataset) -> {model: score}
    # A dataset may appear multiple times across episode configs; we take the
    # first one encountered (sorted order from discover_all_scores).
    per_dataset: Dict[tuple, Dict[str, float]] = {}

    for (dataset, task, ep_config), model_scores in all_scores.items():
        if episode_params and ep_config != episode_params:
            continue
        if dataset not in dataset_to_cluster:
            continue

        cluster_name = dataset_to_cluster[dataset]
        key = (cluster_name, task, dataset)
        if key not in per_dataset:
            per_dataset[key] = {}
        per_dataset[key].update(model_scores)

    # Group by (cluster, task) to find all models and datasets
    cluster_task_groups: Dict[tuple, Dict[str, Dict[str, float]]] = {}
    for (cluster_name, task, dataset), model_scores in per_dataset.items():
        group_key = (cluster_name, task)
        cluster_task_groups.setdefault(group_key, {})
        cluster_task_groups[group_key][dataset] = model_scores

    # Apply complete_datasets filter: within each (cluster, task), drop
    # datasets that not all models have results for.
    if complete_datasets:
        for group_key, dataset_scores in cluster_task_groups.items():
            all_models = set()
            for model_scores in dataset_scores.values():
                all_models.update(model_scores.keys())

            complete = {
                ds: scores for ds, scores in dataset_scores.items()
                if set(scores.keys()) == all_models
            }
            dropped = set(dataset_scores.keys()) - set(complete.keys())
            if dropped:
                cluster_name, task = group_key
                print(f"  Manual cluster '{cluster_name}' / {task}: "
                      f"dropped {len(dropped)} incomplete dataset(s): {sorted(dropped)}")
            cluster_task_groups[group_key] = complete

    # Build tables: one DataFrame per cluster (models x tasks)
    # Also track which datasets ended up in each column.
    cluster_model_task: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    cluster_task_datasets: Dict[str, Dict[str, set]] = {}

    for (cluster_name, task), dataset_scores in cluster_task_groups.items():
        for dataset, model_scores in dataset_scores.items():
            cluster_task_datasets.setdefault(cluster_name, {}).setdefault(task, set()).add(dataset)
            for model, score in model_scores.items():
                cluster_model_task.setdefault(cluster_name, {})
                cluster_model_task[cluster_name].setdefault(model, {})
                cluster_model_task[cluster_name][model].setdefault(task, []).append(score)

    tables: Dict[str, pd.DataFrame] = {}
    column_datasets: Dict[str, Dict[str, List[str]]] = {}
    for cluster_name, model_data in sorted(cluster_model_task.items()):
        records = {}
        for model, task_scores in sorted(model_data.items()):
            records[model] = {
                f"{task} ({TASK_METRICS.get(task, 'score')})": np.mean(scores)
                for task, scores in sorted(task_scores.items())
            }
        df = pd.DataFrame(records).T
        df.index.name = "model"
        tables[cluster_name] = df

        col_ds = {}
        for task in sorted(cluster_task_datasets.get(cluster_name, {})):
            col_name = f"{task} ({TASK_METRICS.get(task, 'score')})"
            col_ds[col_name] = sorted(cluster_task_datasets[cluster_name][task])
        column_datasets[cluster_name] = col_ds

    return tables, column_datasets


def print_manual_cluster_tables(
    tables: Dict[str, pd.DataFrame],
    column_datasets: Dict[str, Dict[str, List[str]]],
    output_dir: str,
) -> None:
    """Print manual cluster tables to stdout and save as markdown.

    Args:
        tables: Mapping from cluster name to DataFrame (models x tasks).
        column_datasets: Mapping from cluster name to dict of column name
            to list of dataset names included in that column.
        output_dir: Directory to save markdown files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for cluster_name, df in tables.items():
        print(f"\n{'='*60}")
        print(f"Manual Cluster — {cluster_name}")
        print(f"{'='*60}\n")

        col_ds = column_datasets.get(cluster_name, {})
        for col_name, datasets in col_ds.items():
            print(f"  {col_name}: {', '.join(datasets)}")
        print()

        print(df.to_markdown())
        print()

        table_path = os.path.join(output_dir, f"manual_clusters_{cluster_name}.md")
        with open(table_path, "w") as f:
            f.write(df.to_markdown() + "\n")
        print(f"Saved: {table_path}")
