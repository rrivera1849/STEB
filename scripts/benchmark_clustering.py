"""Benchmark clustering analysis for STEB.

Discovers which datasets within a task type measure similar constructs
(i.e., rank models the same way) using pairwise Spearman rank correlations
and hierarchical clustering, following the methodology from OLMo 3 Section 3.3.1.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# Add project root so we can import steb
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steb.core import get_supported_datasets
from steb.utils import RESULTS_DIR

# Maps CLI task name -> (path task name, primary metric)
TASK_CONFIG: Dict[str, Tuple[str, str]] = {
    "clustering": ("clustering", "v_measure"),
    "all_to_all_pair_classification": ("pair_classification", "auc"),
    "pre_defined_pair_classification": ("pair_classification", "auc"),
    "order_alignment": ("order_alignment", "distractor_acc_mean"),
    "retrieval": ("retrieval", "mrr"),
    "probing": ("probing", "average"),
}

LOW_CONFIDENCE_THRESHOLD = 10


def discover_scores(
    results_dir: str,
    task_name: str,
    episode_params: Optional[str],
) -> pd.DataFrame:
    """Scan the results directory and build a models x datasets score matrix.

    Args:
        results_dir: Path to the root results directory.
        task_name: The CLI task name (e.g. 'clustering').
        episode_params: Episode params filter like '1_50'. If None, picks the
            first episode params found per dataset-model pair.

    Returns:
        A DataFrame with models as rows and datasets as columns, containing
        the primary metric score for each (model, dataset) pair.
    """
    path_task_name, primary_metric = TASK_CONFIG[task_name]
    supported_datasets = set(get_supported_datasets(task_name))

    scores: Dict[Tuple[str, str], float] = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        return pd.DataFrame()

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        if dataset_name not in supported_datasets:
            continue

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for ep_dir in sorted(model_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue

                if episode_params and ep_dir.name != episode_params:
                    continue

                metrics_file = ep_dir / path_task_name / "metrics.json"
                if not metrics_file.exists():
                    continue

                with open(metrics_file) as f:
                    metrics = json.load(f)

                if primary_metric not in metrics:
                    continue

                scores[(model_name, dataset_name)] = metrics[primary_metric]
                break  # Take first matching episode params

    if not scores:
        return pd.DataFrame()

    models = sorted({m for m, _ in scores})
    datasets = sorted({d for _, d in scores})

    data = {
        d: [scores.get((m, d), np.nan) for m in models]
        for d in datasets
    }
    return pd.DataFrame(data, index=models)


def filter_complete_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Drop models (rows) that don't have results for all datasets.

    Args:
        df: The raw score matrix with models as rows, datasets as columns.

    Returns:
        A DataFrame with only complete rows (no NaN values).
    """
    return df.dropna(axis=0, how="any")


def compute_correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Spearman rank correlations between dataset columns.

    Args:
        df: Score matrix with models as rows and datasets as columns.

    Returns:
        A symmetric DataFrame of pairwise Spearman correlations.
    """
    n_datasets = len(df.columns)
    corr = np.ones((n_datasets, n_datasets))

    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            rho, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            # NaN can occur with constant columns or single-row data
            if np.isnan(rho):
                rho = 0.0
            corr[i, j] = rho
            corr[j, i] = rho

    return pd.DataFrame(corr, index=df.columns, columns=df.columns)


def plot_dendrogram(
    corr_matrix: pd.DataFrame,
    task_name: str,
    output_path: str,
) -> np.ndarray:
    """Plot and save a dendrogram from the correlation matrix.

    Args:
        corr_matrix: Pairwise Spearman correlation matrix.
        task_name: Task name for the plot title.
        output_path: File path to save the figure.

    Returns:
        The linkage matrix from hierarchical clustering.
    """
    # Convert correlation to distance (1 - corr), clip to avoid negative values
    dist = np.clip(1 - corr_matrix.values, 0, 2)
    np.fill_diagonal(dist, 0)
    # Ensure perfect symmetry after floating-point operations
    dist = (dist + dist.T) / 2
    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix) * 0.8), 6))
    dendrogram(
        Z,
        labels=corr_matrix.columns.tolist(),
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=9,
    )
    ax.set_title(f"Dataset Clustering — {task_name}")
    ax.set_ylabel("Distance (1 − Spearman ρ)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return Z


def plot_heatmap(
    corr_matrix: pd.DataFrame,
    Z: np.ndarray,
    task_name: str,
    output_path: str,
) -> None:
    """Plot and save a heatmap of the correlation matrix ordered by clustering.

    Args:
        corr_matrix: Pairwise Spearman correlation matrix.
        Z: Linkage matrix from hierarchical clustering.
        task_name: Task name for the plot title.
        output_path: File path to save the figure.
    """
    # Reorder by dendrogram leaf order
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)
    ordered_labels = [corr_matrix.columns[i] for i in order]
    ordered_corr = corr_matrix.loc[ordered_labels, ordered_labels]

    n = len(ordered_labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)))
    im = ax.imshow(ordered_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ordered_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(ordered_labels, fontsize=8)
    ax.set_title(f"Spearman Correlation — {task_name}")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = ordered_corr.values[i, j]
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_summary(
    Z: np.ndarray,
    corr_matrix: pd.DataFrame,
    n_models: int,
    threshold: float = 1.0,
) -> dict:
    """Build a summary of discovered clusters at a given distance threshold.

    Args:
        Z: Linkage matrix from hierarchical clustering.
        corr_matrix: Pairwise Spearman correlation matrix.
        n_models: Number of models with complete results.
        threshold: Distance threshold for forming flat clusters.

    Returns:
        A dictionary with cluster assignments, metadata, and confidence flag.
    """
    labels = fcluster(Z, t=threshold, criterion="distance")
    datasets = corr_matrix.columns.tolist()

    clusters: Dict[str, List[str]] = {}
    for dataset, label in zip(datasets, labels):
        key = f"cluster_{label}"
        clusters.setdefault(key, []).append(dataset)

    return {
        "n_models": n_models,
        "n_datasets": len(datasets),
        "threshold": threshold,
        "n_clusters": len(clusters),
        "clusters": clusters,
        "low_confidence": n_models < LOW_CONFIDENCE_THRESHOLD,
    }


def aggregate_scores(
    df: pd.DataFrame,
    clusters: Dict[str, List[str]],
) -> pd.DataFrame:
    """Compute cluster-aware aggregated scores for each model.

    For each model: macro-average scores within each cluster, then average
    across clusters. This gives equal weight to each cluster regardless of
    how many datasets it contains.

    Args:
        df: Score matrix with models as rows and datasets as columns.
        clusters: Dict mapping cluster names to lists of dataset names.

    Returns:
        A DataFrame with columns: one per cluster (cluster mean), plus
        'task_score' (the final aggregated score). Indexed by model name.
    """
    result = pd.DataFrame(index=df.index)

    for cluster_name, datasets in sorted(clusters.items()):
        present = [d for d in datasets if d in df.columns]
        if present:
            result[cluster_name] = df[present].mean(axis=1)

    result["task_score"] = result.mean(axis=1)
    return result


def analyze_task(
    results_dir: str,
    task_name: str,
    episode_params: Optional[str],
    output_dir: str,
    min_models: int,
    metric: Optional[str],
) -> bool:
    """Run the full clustering analysis for a single task.

    Args:
        results_dir: Path to the root results directory.
        task_name: The CLI task name.
        episode_params: Episode params filter (e.g. '1_50').
        output_dir: Directory to write output files.
        min_models: Minimum number of models with complete results.
        metric: Override for the primary metric. If None, uses the default.

    Returns:
        True if analysis was produced, False if skipped.
    """
    if metric:
        path_task_name = TASK_CONFIG[task_name][0]
        TASK_CONFIG[task_name] = (path_task_name, metric)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    df = discover_scores(results_dir, task_name, episode_params)
    if df.empty:
        print(f"  No results found. Skipping.")
        return False

    print(f"  Raw matrix: {len(df)} models × {len(df.columns)} datasets")

    df = filter_complete_rows(df)
    if len(df) < min_models:
        print(f"  Only {len(df)} models with complete results (need {min_models}). Skipping.")
        return False

    print(f"  Complete matrix: {len(df)} models × {len(df.columns)} datasets")

    if len(df.columns) < 2:
        print(f"  Only {len(df.columns)} dataset(s). Nothing to cluster. Skipping.")
        return False

    if len(df) < LOW_CONFIDENCE_THRESHOLD:
        print(f"  Warning: fewer than {LOW_CONFIDENCE_THRESHOLD} models — results are low-confidence.")

    os.makedirs(output_dir, exist_ok=True)

    # Save score matrix
    scores_path = os.path.join(output_dir, f"{task_name}_scores.csv")
    df.to_csv(scores_path)
    print(f"  Saved score matrix: {scores_path}")

    # Compute correlations
    corr = compute_correlation_matrix(df)
    corr_path = os.path.join(output_dir, f"{task_name}_correlation.csv")
    corr.to_csv(corr_path)
    print(f"  Saved correlation matrix: {corr_path}")

    # Dendrogram
    dendro_path = os.path.join(output_dir, f"{task_name}_dendrogram.png")
    Z = plot_dendrogram(corr, task_name, dendro_path)
    print(f"  Saved dendrogram: {dendro_path}")

    # Heatmap
    heatmap_path = os.path.join(output_dir, f"{task_name}_heatmap.png")
    plot_heatmap(corr, Z, task_name, heatmap_path)
    print(f"  Saved heatmap: {heatmap_path}")

    # Summary
    summary = build_summary(Z, corr, n_models=len(df))
    summary_path = os.path.join(output_dir, f"{task_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")
    print(f"  Clusters (threshold={summary['threshold']}):")
    for name, members in summary["clusters"].items():
        print(f"    {name}: {members}")

    # Aggregated scores: mean within clusters, then mean across clusters
    agg = aggregate_scores(df, summary["clusters"])
    agg_path = os.path.join(output_dir, f"{task_name}_aggregated.csv")
    agg.to_csv(agg_path)
    print(f"  Saved aggregated scores: {agg_path}")
    print(f"  Task scores (cluster-aware):")
    for model in agg.index:
        print(f"    {model}: {agg.loc[model, 'task_score']:.4f}")

    return True


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
        choices=list(TASK_CONFIG.keys()),
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
    return parser.parse_args()


def main() -> None:
    """Entry point for benchmark clustering analysis."""
    args = parse_args()

    if not args.task and not args.all_tasks:
        print("Error: specify --task <name> or --all-tasks.")
        sys.exit(1)

    tasks = list(TASK_CONFIG.keys()) if args.all_tasks else [args.task]
    produced = 0

    for task in tasks:
        if analyze_task(
            args.results_dir,
            task,
            args.episode_params,
            args.output_dir,
            args.min_models,
            args.metric if not args.all_tasks else None,
        ):
            produced += 1

    print(f"\nDone. Produced analysis for {produced}/{len(tasks)} tasks.")


if __name__ == "__main__":
    main()
