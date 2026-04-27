"""Manual cluster tables built from a YAML cluster definition file.

Each YAML entry is a dataset name optionally followed by per-entry
flags. Two flags are supported, in any order:

    NAME --oa_variant {distractor,acc}    select metric for order_alignment
    NAME --oa_only                        restrict the entry to order_alignment

`--oa_variant` only changes which metric is used for the order_alignment
task; the entry still contributes to all other tasks the dataset
declares. `--oa_only` is the separate task-restriction switch. Either
flag may appear without the other.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from steb.presets import get_benchmark_config

from .config import ClusterEntry, OA_VARIANT_METRICS, TASK_METRICS
from .discovery import (
    _resolve_metric_for_entry,
    _warn_missing_metric,
    discover_all_scores,
)


def _benchmark_default_ep_configs() -> Dict[str, str]:
    """Return ``{task: canonical_ep_config_string}`` for the benchmark preset.

    The episode-config string is ``"{episode_size}_{n_episodes_per_class}"``,
    matching the directory layout produced by ``steb run``. When a task
    has multiple episode_sizes in the preset, the first one is taken.
    Used by ``build_manual_cluster_tables`` to pick a deterministic
    ep_config per task when ``--episode-params`` isn't passed.
    """
    defaults: Dict[str, str] = {}
    config = get_benchmark_config()
    for item in config["config"]["tasks"]:
        ep_size = item["episode_sizes"][0]
        n_eps = item["n_episodes_per_class"]
        defaults[item["task"]] = f"{ep_size}_{n_eps}"
    return defaults


def parse_cluster_entry(
    entry: str,
) -> ClusterEntry:
    """Parse a cluster YAML dataset entry into a ClusterEntry.

    Supports an optional --oa_variant VALUE flag and an optional
    --oa_only flag, in any order.

    Args:
        entry: A raw entry string from the cluster YAML.

    Returns:
        A ClusterEntry capturing the dataset name and any flags.
    """
    name, *tokens = entry.split()
    oa_variant: Optional[str] = None
    oa_only = False
    it = iter(tokens)
    for tok in it:
        if tok == "--oa_only":
            oa_only = True
        elif tok == "--oa_variant":
            oa_variant = next(it, None)
            if oa_variant not in OA_VARIANT_METRICS:
                raise ValueError(
                    f"Invalid cluster entry '{entry}': --oa_variant expected "
                    f"one of {sorted(OA_VARIANT_METRICS)}, got {oa_variant!r}."
                )
        else:
            raise ValueError(
                f"Invalid cluster entry '{entry}': unexpected token {tok!r}."
            )
    return ClusterEntry(name=name, oa_variant=oa_variant, oa_only=oa_only)


def load_manual_clusters(
    clusters_path: str,
) -> Dict[str, List[ClusterEntry]]:
    """Load manual dataset clusters from a YAML file.

    Args:
        clusters_path: Path to the YAML file with cluster definitions.

    Returns:
        A dict mapping cluster name to a list of ClusterEntry records.
    """
    with open(clusters_path) as f:
        raw = yaml.safe_load(f)

    clusters: Dict[str, List[ClusterEntry]] = {}
    for name, config in raw.items():
        clusters[name] = [parse_cluster_entry(entry) for entry in config["datasets"]]
    return clusters


def build_manual_cluster_tables(
    results_dir: str,
    clusters: Dict[str, List[ClusterEntry]],
    episode_params: Optional[str],
    include_excluded: bool = False,
    complete_datasets: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, List[str]]]]:
    """Build one table per manual cluster.

    For each cluster, produces a DataFrame whose rows are models and
    columns are keyed by (task, metric) — since order_alignment can
    appear under different metrics depending on each entry's --oa_variant,
    the metric is part of the column key. Each cell is the average metric
    across datasets in that cluster that contribute to the column.

    Args:
        results_dir: Path to the root results directory.
        clusters: Mapping from cluster name to a list of ClusterEntry
            records.
        episode_params: Episode params filter (e.g. '1_50'). If None, the
            per-task canonical ep_config from the benchmark preset is used
            for each task (so manual cluster tables stay deterministic
            instead of silently mixing every ep_config that exists on
            disk).
        include_excluded: If True, include semantic and non-English datasets.
        complete_datasets: If True, within each cluster column (a
            ``task (metric)`` slot), drop datasets that not all models
            have results for.

    Returns:
        A tuple of:
          - A dict mapping cluster name to a DataFrame whose rows are
            models and whose columns are keyed by (task, metric).
          - A dict mapping cluster name to a dict of column name to
            list of dataset names included in that column.
    """
    all_scores = discover_all_scores(results_dir, include_excluded)
    if not all_scores:
        return {}, {}

    # Invert clusters: dataset -> list of (cluster_name, entry). A dataset
    # may legitimately appear in more than one cluster (e.g. once plain in
    # 'style', once with --oa_only in 'style_vs_content').
    dataset_to_entries: Dict[str, List[Tuple[str, ClusterEntry]]] = {}
    for cluster_name, entries in clusters.items():
        for entry in entries:
            dataset_to_entries.setdefault(entry.name, []).append((cluster_name, entry))

    # When the user didn't pass --episode-params, fall back to the
    # benchmark preset's canonical ep_config per task so the table stays
    # deterministic instead of mixing every ep_config on disk.
    benchmark_defaults = None if episode_params else _benchmark_default_ep_configs()

    if episode_params:
        print(f"Manual clusters: using --episode-params {episode_params!r} for every task.")
    else:
        print("Manual clusters: --episode-params not set; using benchmark preset defaults per task:")
        for task, ep_config in sorted(benchmark_defaults.items()):
            print(f"  {task}: {ep_config}")

    # Walk discovered runs, honour each entry's --oa_only and --oa_variant
    # flags, and bucket scores by (cluster, task, metric, dataset).
    scores_by_cluster_task_metric: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]] = {}
    warned_missing: set = set()

    for (dataset, task, ep_config), model_metrics in all_scores.items():
        if episode_params:
            if ep_config != episode_params:
                continue
        elif benchmark_defaults.get(task) != ep_config:
            continue
        if dataset not in dataset_to_entries:
            continue

        for cluster_name, entry in dataset_to_entries[dataset]:
            if entry.oa_only and task != "order_alignment":
                continue

            metric_key = _resolve_metric_for_entry(task, entry)

            scores: Dict[str, float] = {}
            for model, metrics in model_metrics.items():
                value = metrics.get(metric_key)
                if value is None:
                    _warn_missing_metric(dataset, task, metric_key, warned_missing)
                    continue
                scores[model] = value

            if not scores:
                continue

            key = (cluster_name, task, metric_key)
            scores_by_cluster_task_metric.setdefault(key, {}).setdefault(dataset, {}).update(scores)

    # Warn for any cluster entry whose dataset never produced a results
    # row (typo in the YAML, missing results dir, or filtered out by
    # --episode-params).
    existing_datasets = {dataset for (dataset, _, _) in all_scores}
    for cluster_name, entries in clusters.items():
        for entry in entries:
            if entry.name not in existing_datasets:
                print(
                    f"  WARNING: cluster '{cluster_name}' entry "
                    f"'{entry.name}': no results found.",
                    file=sys.stderr,
                )

    # Apply complete_datasets filter: within each cluster column (a
    # `task (metric)` slot), drop datasets that not all models have
    # results for.
    if complete_datasets:
        for key, dataset_scores in scores_by_cluster_task_metric.items():
            all_models: set[str] = set()
            for model_scores in dataset_scores.values():
                all_models.update(model_scores.keys())

            complete = {
                ds: scores for ds, scores in dataset_scores.items()
                if set(scores.keys()) == all_models
            }
            dropped = set(dataset_scores.keys()) - set(complete.keys())
            if dropped:
                cluster_name, task, metric_key = key
                print(f"  Manual cluster '{cluster_name}' / {task} ({metric_key}): "
                      f"dropped {len(dropped)} incomplete dataset(s): {sorted(dropped)}")
            scores_by_cluster_task_metric[key] = complete

    # Build tables: one DataFrame per cluster, rows = models, columns
    # keyed by (task, metric). Also track which datasets ended up in each column.
    cluster_model_col: Dict[str, Dict[str, Dict[Tuple[str, str], List[float]]]] = {}
    cluster_col_datasets: Dict[str, Dict[Tuple[str, str], set]] = {}

    for (cluster_name, task, metric_key), dataset_scores in scores_by_cluster_task_metric.items():
        col_id = (task, metric_key)
        for dataset, model_scores in dataset_scores.items():
            cluster_col_datasets.setdefault(cluster_name, {}).setdefault(col_id, set()).add(dataset)
            for model, score in model_scores.items():
                cluster_model_col.setdefault(cluster_name, {})
                cluster_model_col[cluster_name].setdefault(model, {})
                cluster_model_col[cluster_name][model].setdefault(col_id, []).append(score)

    tables: Dict[str, pd.DataFrame] = {}
    column_datasets: Dict[str, Dict[str, List[str]]] = {}
    for cluster_name, model_data in sorted(cluster_model_col.items()):
        records = {}
        for model, col_scores in sorted(model_data.items()):
            records[model] = {
                f"{task} ({metric_key})": np.mean(scores)
                for (task, metric_key), scores in sorted(col_scores.items())
            }
        df = pd.DataFrame(records).T
        df.index.name = "model"
        tables[cluster_name] = df

        col_ds = {}
        for col_id in sorted(cluster_col_datasets.get(cluster_name, {})):
            task, metric_key = col_id
            col_ds[f"{task} ({metric_key})"] = sorted(cluster_col_datasets[cluster_name][col_id])
        column_datasets[cluster_name] = col_ds

    return tables, column_datasets


def print_manual_cluster_tables(
    tables: Dict[str, pd.DataFrame],
    column_datasets: Dict[str, Dict[str, List[str]]],
    output_dir: str,
) -> None:
    """Print manual cluster tables to stdout and save as markdown.

    Args:
        tables: Mapping from cluster name to DataFrame whose rows are
            models and whose columns are keyed by (task, metric).
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
