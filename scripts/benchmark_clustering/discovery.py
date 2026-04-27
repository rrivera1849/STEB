"""Read scores from the results directory.

`discover_scores` produces a per-task models × datasets matrix used by
the auto-cluster path. `discover_all_scores` produces a flat dump of
top-level scalar metrics keyed by (dataset, task, episode_config) used
by the manual-cluster and Excel-export paths. Two small consumer
helpers (`_warn_missing_metric`, `_resolve_metric_for_entry`) live
here too because they're shared between those downstream paths.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from steb.core import get_supported_datasets

from .config import (
    ClusterEntry,
    EXCLUDED_DATASETS,
    EXCLUDED_MODELS,
    OA_VARIANT_METRICS,
    TASK_METRICS,
)


def _warn_missing_metric(
    dataset: str,
    task: str,
    metric: str,
    seen: set,
) -> None:
    """Print a deduped warning to stderr when a metric is missing for a run.

    Each (dataset, task, metric) combination is warned about at most once
    per call site (deduped via the caller-provided ``seen`` set), so model
    counts are not part of the dedupe key — a single warning per data
    triple is enough to alert the user.
    """
    key = (dataset, task, metric)
    if key in seen:
        return
    seen.add(key)
    print(
        f"  WARNING: ignoring runs for dataset '{dataset}' / task '{task}' "
        f"that are missing the '{metric}' metric.",
        file=sys.stderr,
    )


def _resolve_metric_for_entry(
    task: str,
    entry: ClusterEntry,
) -> Optional[str]:
    """Pick which metric an entry contributes to a (task, metric) column.

    --oa_variant only matters for the order_alignment task; for all other
    tasks the default TASK_METRICS metric is used. Returns None when the
    task is unknown to TASK_METRICS (so the caller should skip).
    """
    if task == "order_alignment" and entry.oa_variant is not None:
        return OA_VARIANT_METRICS[entry.oa_variant]
    return TASK_METRICS.get(task)


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
) -> Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]]:
    """Scan the results directory and collect top-level metrics across tasks.

    Respects EXCLUDED_DATASETS, EXCLUDED_MODELS, and NON_ENGLISH_DATASETS
    filtering. Collects every (dataset, task, episode_config, model)
    combination found, returning the top-level scalar metrics from each
    run's metrics.json so callers can pick whichever metric they need
    (e.g. acc_mean vs distractor_acc_mean for order_alignment). Nested
    values (e.g. _per_label, submetrics) are dropped to keep memory low.

    Args:
        results_dir: Path to the root results directory.
        include_excluded: If True, include semantic and non-English datasets.

    Returns:
        A dict mapping (dataset, task, episode_config) to a dict mapping
        model name to that run's flat (scalar-only) metrics dict.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return {}

    # Build a map of dataset -> set of tasks it supports
    dataset_tasks: Dict[str, set] = {}
    for task_name in TASK_METRICS:
        for ds in get_supported_datasets(task_name):
            dataset_tasks.setdefault(ds, set()).add(task_name)

    # Collect: (dataset, task, episode_config) -> {model: metrics_dict}
    rows: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}

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
                    metrics_file = ep_dir / task_name / "metrics.json"
                    if not metrics_file.exists():
                        continue

                    with open(metrics_file) as f:
                        metrics = json.load(f)

                    # Strip nested values (e.g. _per_label, submetrics) — only
                    # top-level scalar metrics are needed by current consumers.
                    top_level_metrics = {
                        k: v for k, v in metrics.items() if not isinstance(v, dict)
                    }

                    key = (dataset_name, task_name, ep_dir.name)
                    rows.setdefault(key, {})[model_dir.name] = top_level_metrics

    return rows
