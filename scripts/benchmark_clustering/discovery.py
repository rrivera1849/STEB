"""Read scores from the results directory."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from steb.core import get_supported_datasets

from .config import EXCLUDED_DATASETS, EXCLUDED_MODELS, TASK_METRICS


def discover_scores(
    results_dir: str,
    task_name: str,
    primary_metric: str,
    episode_params: Optional[str],
    include_excluded: bool = False,
) -> pd.DataFrame:
    """Scan the results directory and build a models x datasets score matrix.

    Args:
        results_dir: Path to the root results directory.
        task_name: The CLI task name (e.g. 'clustering').
        primary_metric: The metric to extract from metrics.json.
        episode_params: Episode params filter like '1_50'. If None, picks the
            first episode params found per dataset-model pair.
        include_excluded: If True, include semantic and non-English datasets.

    Returns:
        A DataFrame with models as rows and datasets as columns.
    """
    supported_datasets = set(get_supported_datasets(task_name))
    scores: Dict[str, Dict[str, float]] = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        return pd.DataFrame()

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        if dataset_name not in supported_datasets:
            continue
        if not include_excluded and dataset_name in EXCLUDED_DATASETS:
            continue

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name in EXCLUDED_MODELS:
                continue

            for ep_dir in sorted(model_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                if episode_params and ep_dir.name != episode_params:
                    continue

                metrics_file = ep_dir / task_name / "metrics.json"
                if not metrics_file.exists():
                    continue

                with open(metrics_file) as f:
                    metrics = json.load(f)

                if primary_metric not in metrics:
                    continue

                scores.setdefault(model_dir.name, {})[dataset_name] = metrics[primary_metric]
                break  # Take first matching episode params

    if not scores:
        return pd.DataFrame()

    return pd.DataFrame(scores).T.rename_axis("model")


def discover_all_scores(
    results_dir: str,
    include_excluded: bool = False,
) -> List[Dict[str, object]]:
    """Scan the results directory and collect all scores across tasks.

    Respects EXCLUDED_DATASETS, EXCLUDED_MODELS, and NON_ENGLISH_DATASETS
    filtering. Collects every (dataset, task, episode_config, model, metric)
    combination found.

    Args:
        results_dir: Path to the root results directory.
        include_excluded: If True, include semantic and non-English datasets.

    Returns:
        A list of row dicts with keys: dataset, task, episode_config,
        primary_metric, and one key per model.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    # Build a map of dataset -> set of tasks it supports
    dataset_tasks: Dict[str, set] = {}
    for task_name in TASK_METRICS:
        for ds in get_supported_datasets(task_name):
            dataset_tasks.setdefault(ds, set()).add(task_name)

    # Collect: (dataset, task, episode_config) -> {model: score}
    rows: Dict[tuple, Dict[str, float]] = {}

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        if dataset_name not in dataset_tasks:
            continue
        if not include_excluded and dataset_name in EXCLUDED_DATASETS:
            continue

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name in EXCLUDED_MODELS:
                continue

            for ep_dir in sorted(model_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue

                for task_name in dataset_tasks[dataset_name]:
                    metric_key = TASK_METRICS[task_name]
                    metrics_file = ep_dir / task_name / "metrics.json"
                    if not metrics_file.exists():
                        continue

                    with open(metrics_file) as f:
                        metrics = json.load(f)

                    if metric_key not in metrics:
                        continue

                    key = (dataset_name, task_name, ep_dir.name)
                    rows.setdefault(key, {})[model_dir.name] = metrics[metric_key]

    return rows
