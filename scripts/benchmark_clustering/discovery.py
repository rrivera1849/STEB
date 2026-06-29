"""Read scores from the results directory.

`discover_scores` produces a per-task models × datasets matrix used by
the auto-cluster path. `discover_all_scores` produces a flat dump of
top-level scalar metrics keyed by (dataset, task, episode_config) used
by the manual-cluster and Excel-export paths. Two small consumer
helpers (`_warn_missing_metric`, `_resolve_metric_for_entry`) live
here too because they're shared between those downstream paths.

Both discovery functions also scan a sibling ``submitted_results/`` tree
at the repository root when it exists, merging community submissions
into the same matrix. Maintainer-owned results win on collisions, so
``submitted_results/`` can only add coverage, never overwrite it.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from steb.core import get_supported_datasets

from .config import (
    ClusterEntry,
    EXCLUDED_DATASETS,
    EXCLUDED_MODELS,
    OA_VARIANT_METRICS,
    TASK_METRICS,
)

# Location of the community-submitted results tree, relative to the repo
# root. ``benchmark_clustering`` auto-ingests this directory when it
# exists; see the "Submitting your model" section of the project README
# for the contribution flow.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUBMITTED_RESULTS_DIR = _PROJECT_ROOT / "submitted_results"


def _results_dirs_to_scan(results_dir: str) -> List[Path]:
    """Return the ordered list of roots discovery should scan.

    The maintainer-supplied ``results_dir`` comes first so it wins on any
    (dataset, model, episode_config, task) collision with the
    community-submitted tree.
    """
    roots = [Path(results_dir)]
    if _SUBMITTED_RESULTS_DIR.exists() and _SUBMITTED_RESULTS_DIR != roots[0]:
        roots.append(_SUBMITTED_RESULTS_DIR)
    return roots


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
) -> str:
    """Pick which metric an entry contributes to a (task, metric) column.

    --oa_variant only matters for the order_alignment task; for all other
    tasks the default TASK_METRICS metric is used. Asserts that ``task``
    is a key of ``TASK_METRICS`` — every task fed in here comes from
    discovery, which only ever yields tasks that are in ``TASK_METRICS``.
    """
    if task == "order_alignment" and entry.oa_variant is not None:
        return OA_VARIANT_METRICS[entry.oa_variant]
    metric = TASK_METRICS.get(task)
    assert metric is not None, f"Unknown task {task!r} (not in TASK_METRICS)"
    return metric

#   This is for the Auto Clustering Path
def discover_scores(
    results_dir: str,
    task_name: str,
    primary_metric: str,
    episode_params: Optional[str],
    include_excluded: bool = False,
    allowed_models: Optional[set] = None,
) -> pd.DataFrame:
    """Scan the results directory and build a models x datasets score matrix.

    Args:
        results_dir: Path to the root results directory.
        task_name: The CLI task name (e.g. 'clustering').
        primary_metric: The metric to extract from metrics.json.
        episode_params: Episode params filter like '1_50'. If the value
            ends with '_' (e.g. '1_') it is treated as a prefix filter on
            episode size. If None, picks the first episode params found
            per dataset-model pair.
        include_excluded: If True, include semantic and non-English datasets.
        allowed_models: If provided, only include models in this set.

    Returns:
        A DataFrame with models as rows and datasets as columns.
    """
    supported_datasets = set(get_supported_datasets(task_name))
    scores: Dict[str, Dict[str, float]] = {}

    for results_path in _results_dirs_to_scan(results_dir):
        if not results_path.exists():
            continue

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
                if allowed_models is not None and model_dir.name not in allowed_models:
                    continue

                # First-write-wins on collision: a maintainer-supplied
                # (model, dataset) score is not overwritten by a later root.
                if scores.get(model_dir.name, {}).get(dataset_name) is not None:
                    continue

                for ep_dir in sorted(model_dir.iterdir()):
                    if not ep_dir.is_dir():
                        continue
                    if episode_params:
                        if episode_params.endswith("_"):
                            if not ep_dir.name.startswith(episode_params):
                                continue
                        elif ep_dir.name != episode_params:
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

#   This is for the Manual Clustering Path
def discover_all_scores(
    results_dir: str,
    include_excluded: bool = False,
    allowed_models: Optional[set] = None,
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
        allowed_models: If provided, only include models in this set.

    Returns:
        A dict mapping (dataset, task, episode_config) to a dict mapping
        model name to that run's flat (scalar-only) metrics dict.
    """
    # Build a map of dataset -> set of tasks it supports
    dataset_tasks: Dict[str, set] = {}
    for task_name in TASK_METRICS:
        for ds in get_supported_datasets(task_name):
            dataset_tasks.setdefault(ds, set()).add(task_name)

    # Collect: (dataset, task, episode_config) -> {model: metrics_dict}
    rows: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}

    for results_path in _results_dirs_to_scan(results_dir):
        if not results_path.exists():
            continue

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
                if allowed_models is not None and model_dir.name not in allowed_models:
                    continue

                for ep_dir in sorted(model_dir.iterdir()):
                    if not ep_dir.is_dir():
                        continue

                    for task_name in dataset_tasks[dataset_name]:
                        metrics_file = ep_dir / task_name / "metrics.json"
                        if not metrics_file.exists():
                            continue

                        key = (dataset_name, task_name, ep_dir.name)
                        # First-write-wins: a maintainer-supplied score for
                        # this (key, model) is not overwritten by a later root.
                        if rows.get(key, {}).get(model_dir.name) is not None:
                            continue

                        with open(metrics_file) as f:
                            metrics = json.load(f)

                        # Strip nested values (e.g. _per_label, submetrics) — only
                        # top-level scalar metrics are needed by current consumers.
                        top_level_metrics = {
                            k: v for k, v in metrics.items() if not isinstance(v, dict)
                        }

                        rows.setdefault(key, {})[model_dir.name] = top_level_metrics

    return rows
