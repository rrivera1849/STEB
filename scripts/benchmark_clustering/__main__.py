"""CLI entry point for benchmark_clustering.

Run with:
    python -m scripts.benchmark_clustering [args]
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure the project root is on sys.path so 'from steb.utils import ...'
# works regardless of the directory the user invokes this from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project-relative defaults that work regardless of the invoking CWD.
_DEFAULT_MANUAL_CLUSTERS = str(_PROJECT_ROOT / "scripts" / "dataset_clusters.yaml")
_DEFAULT_MODELS_FILE = str(_PROJECT_ROOT / "scripts" / "models_all.txt")
_DEFAULT_EXPORT_EXCEL = "scores.xlsx"
_DEFAULT_THRESHOLD = 0.5

import pandas as pd

from steb.utils import RESULTS_DIR

from .auto_cluster import analyze_task, plot_model_ranking, print_summary_table
from .config import TASK_METRICS
from .excel_export import export_excel
from .manual_cluster import (
    build_manual_cluster_tables,
    load_manual_clusters,
    print_manual_cluster_tables,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark clustering analysis for STEB tasks.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Path to the results directory (default: %(default)s).",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_METRICS.keys()),
        help="Analyze only this task. When omitted, all tasks run unless --no-all-tasks is set.",
    )
    parser.add_argument(
        "--all-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run analysis for all task types (default: on). Pass --no-all-tasks to skip "
             "task analysis entirely (e.g. for manual-clusters-only or episode-only runs). "
             "Overridden by --task <name> when both are present.",
    )
    parser.add_argument(
        "--metric",
        help="Override the primary metric (default: task-specific).",
    )
    parser.add_argument(
        "--episode-params",
        help="Filter to a specific episode params string (e.g. '1_50').",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Directory for output files (default: %(default)s).",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=3,
        help="Minimum models with complete results to run analysis (default: %(default)s).",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include semantic and non-English datasets that are excluded by default.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help="Distance threshold for flat clustering (default: %(default)s). "
             "Lower = more clusters (0.5 ≈ ρ≥0.5), higher = fewer (1.5 ≈ ρ≥-0.5).",
    )
    parser.add_argument(
        "--export-excel",
        metavar="PATH",
        default=_DEFAULT_EXPORT_EXCEL,
        help="Export all scores to an Excel file (default: %(default)s). Sheet 1 has "
             "per-dataset scores, 'summary' is the Operational STEB score, "
             "'STEB_definitional' is the Definitional STEB score.",
    )
    parser.add_argument(
        "--manual-clusters",
        metavar="PATH",
        default=_DEFAULT_MANUAL_CLUSTERS,
        help="Path to a YAML file defining manual dataset clusters "
             "(default: scripts/dataset_clusters.yaml).",
    )
    parser.add_argument(
        "--models-file",
        metavar="PATH",
        default=_DEFAULT_MODELS_FILE,
        help="Path to a models file, one org/model per line "
             "(default: scripts/models_all.txt).",
    )
    return parser.parse_args()


def parse_models_file(
    models_file: str,
) -> Set[str]:
    """Parse a models file and return the set of short model names.

    Each non-blank, non-comment line is expected to be a model identifier
    like 'org/model-name'. The short name is the part after the last '/'.

    Args:
        models_file: Path to the models file.

    Returns:
        Set of short model names.
    """
    models = set()
    with open(models_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.add(line.split("/")[-1])
    return models


def main() -> None:
    """Entry point for benchmark clustering analysis."""
    args = parse_args()

    # `--complete-datasets` and `--mc-complete-datasets` are no longer optional:
    # benchmark clustering only operates with complete-dataset filtering on, both
    # for auto-discovered clusters and for manual clusters. The flags were
    # removed from the CLI; argparse will hard-error on the old spellings.
    complete_datasets = True
    mc_complete_datasets = True

    allowed_models: Optional[Set[str]] = None
    if args.models_file:
        allowed_models = parse_models_file(args.models_file)
        print(f"Filtering to {len(allowed_models)} models from {args.models_file}")

    # Task selection: explicit --task <name> wins; otherwise --all-tasks
    # (default on) runs every task; --no-all-tasks with no --task skips
    # task analysis entirely.
    if args.task:
        tasks: List[str] = [args.task]
    elif args.all_tasks:
        tasks = list(TASK_METRICS.keys())
    else:
        tasks = []

    task_scores: Dict[str, pd.Series] = {}
    task_n_datasets: Dict[str, int] = {}
    effective_metrics: Dict[str, str] = {}
    ranking_plot_paths: Optional[List[str]] = None

    if tasks:
        is_single_task = args.task is not None and len(tasks) == 1
        for task in tasks:
            metric = args.metric if is_single_task else TASK_METRICS[task]
            if metric is None:
                metric = TASK_METRICS[task]
            effective_metrics[task] = metric

            result = analyze_task(
                args.results_dir,
                task,
                metric,
                args.episode_params,
                args.output_dir,
                args.min_models,
                args.include_excluded,
                args.threshold,
                complete_datasets,
                allowed_models,
            )
            if result is not None:
                scores, n_datasets = result
                task_scores[task] = scores
                task_n_datasets[task] = n_datasets

        # Compute an overall STEB average across tasks for each model
        if len(task_scores) > 1:
            all_task_df = pd.DataFrame(task_scores)
            task_scores["STEB_score"] = all_task_df.mean(axis=1)
            effective_metrics["STEB_score"] = "avg"
            task_n_datasets["STEB_score"] = sum(task_n_datasets.values())

        print_summary_table(task_scores, effective_metrics, task_n_datasets, args.output_dir)

        # Plot model ranking if we have the STEB score
        if "STEB_score" in task_scores:
            ranking_plot_paths = plot_model_ranking(task_scores["STEB_score"], args.output_dir)

        print(f"Done. Produced analysis for {len(task_scores)}/{len(tasks)} tasks.")

    manual_cluster_tables: Optional[Dict[str, pd.DataFrame]] = None
    manual_cluster_datasets: Optional[Dict[str, Dict[str, List[str]]]] = None
    if args.manual_clusters:
        try:
            clusters = load_manual_clusters(args.manual_clusters)
        except ValueError as e:
            print(f"error loading {args.manual_clusters}: {e}", file=sys.stderr)
            sys.exit(2)
        manual_cluster_tables, manual_cluster_datasets = build_manual_cluster_tables(
            args.results_dir,
            clusters,
            args.episode_params,
            args.include_excluded,
            mc_complete_datasets,
            allowed_models,
        )
        print_manual_cluster_tables(manual_cluster_tables, manual_cluster_datasets, args.output_dir)

    if args.export_excel:
        export_excel(
            args.results_dir,
            args.export_excel,
            args.include_excluded,
            task_scores if task_scores else None,
            effective_metrics if effective_metrics else None,
            task_n_datasets if task_n_datasets else None,
            manual_cluster_tables,
            manual_cluster_datasets,
            ranking_plot_paths=ranking_plot_paths if task_scores else None,
            allowed_models=allowed_models,
        )


if __name__ == "__main__":
    main()
