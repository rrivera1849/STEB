"""CLI entry point for benchmark_clustering.

Run with:
    python -m scripts.benchmark_clustering [args]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the project root is on sys.path so 'from steb.utils import ...'
# works regardless of the directory the user invokes this from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from steb.utils import RESULTS_DIR

from .auto_cluster import analyze_task, print_summary_table
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
        help="Task to analyze. If omitted with --all-tasks, analyzes all.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run analysis for all task types.",
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
        default=1.0,
        help="Distance threshold for flat clustering (default: %(default)s). "
             "Lower = more clusters (0.5 ≈ ρ≥0.5), higher = fewer (1.5 ≈ ρ≥-0.5).",
    )
    parser.add_argument(
        "--complete-datasets",
        action="store_true",
        help="Instead of dropping models with missing datasets, drop datasets "
             "that not all models have results for. Keeps all models.",
    )
    parser.add_argument(
        "--export-excel",
        metavar="PATH",
        help="Export all scores to an Excel file. Sheet 1 has per-dataset "
             "scores, sheet 2 has the cluster-aware summary (if --task or "
             "--all-tasks is also provided).",
    )
    parser.add_argument(
        "--manual-clusters",
        metavar="PATH",
        help="Path to a YAML file defining manual dataset clusters. "
             "Produces one table per cluster showing average scores per task.",
    )
    parser.add_argument(
        "--mc-complete-datasets",
        action="store_true",
        help="For manual clusters: within each (cluster, task) group, drop "
             "datasets that not all models have results for.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for benchmark clustering analysis."""
    args = parse_args()

    if not args.task and not args.all_tasks and not args.export_excel and not args.manual_clusters:
        print("Error: specify --task <name>, --all-tasks, --export-excel, or --manual-clusters.")
        sys.exit(1)

    task_scores: Dict[str, pd.Series] = {}
    effective_metrics: Dict[str, str] = {}

    if args.task or args.all_tasks:
        tasks = list(TASK_METRICS.keys()) if args.all_tasks else [args.task]

        for task in tasks:
            metric = args.metric if not args.all_tasks else TASK_METRICS[task]
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
                args.complete_datasets,
            )
            if result is not None:
                task_scores[task] = result

        print_summary_table(task_scores, effective_metrics, args.output_dir)
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
            args.mc_complete_datasets,
        )
        print_manual_cluster_tables(manual_cluster_tables, manual_cluster_datasets, args.output_dir)

    if args.export_excel:
        export_excel(
            args.results_dir,
            args.export_excel,
            args.include_excluded,
            task_scores if task_scores else None,
            effective_metrics if effective_metrics else None,
            manual_cluster_tables,
            manual_cluster_datasets,
        )


if __name__ == "__main__":
    main()
