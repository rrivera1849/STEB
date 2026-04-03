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
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# Add project root so we can import steb
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steb.core import get_supported_datasets
from steb.utils import RESULTS_DIR

# Maps task name -> primary metric
TASK_METRICS: Dict[str, str] = {
    "clustering": "v_measure",
    "all_to_all_pair_classification": "auc",
    "pre_defined_pair_classification": "auc",
    "order_alignment": "distractor_acc_mean",
    "retrieval": "mrr",
    "probing": "average",
}

# Datasets where the label is primarily semantic (topic, sentiment, content)
# rather than stylistic. Excluded from analysis by default.
SEMANTIC_DATASETS = {
    # Topic / content-based
    "20_Newsgroups_Fixed",
    "ag_news",
    "reuters21578",
    # Sentiment / emotion
    "emotion",
    "financial_phrasebank",
    "twitter-airline-sentiment",
    "yelp_polarity",
    # MISC (not semantic, but I don't want it)
    "probing_blog_small",
}

# Non-English datasets. Excluded from analysis by default.
NON_ENGLISH_DATASETS = {
    # PAN13
    "pan13_authorship_verification_greek_test",
    "pan13_authorship_verification_spanish_test",
    # PAN14
    "pan14_authorship_verification_corpus1_dutch_essays_test",
    "pan14_authorship_verification_corpus1_dutch_reviews_test",
    "pan14_authorship_verification_corpus1_greek_articles_test",
    "pan14_authorship_verification_corpus1_spanish_articles_test",
    "pan14_authorship_verification_corpus2_dutch_essays_test",
    "pan14_authorship_verification_corpus2_dutch_reviews_test",
    "pan14_authorship_verification_corpus2_greek_articles_test",
    "pan14_authorship_verification_corpus2_spanish_articles_test",
    # PAN15
    "pan15_authorship_verification_dutch_test",
    "pan15_authorship_verification_greek_test",
    "pan15_authorship_verification_spanish_test",
    # PAN18
    "pan18_cross_domain_authorship_attribution_french",
    "pan18_cross_domain_authorship_attribution_italian",
    "pan18_cross_domain_authorship_attribution_polish",
    "pan18_cross_domain_authorship_attribution_spanish",
}

EXCLUDED_DATASETS = SEMANTIC_DATASETS | NON_ENGLISH_DATASETS

LOW_CONFIDENCE_THRESHOLD = 10


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


def compute_correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise Spearman rank correlations between dataset columns.

    Args:
        df: Score matrix with models as rows and datasets as columns.

    Returns:
        A symmetric DataFrame of pairwise Spearman correlations.
    """
    n = len(df.columns)
    corr = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            if np.isnan(rho):
                rho = 0.0
            corr[i, j] = corr[j, i] = rho

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
    dist = np.clip(1 - corr_matrix.values, 0, 2)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist), method="ward")

    fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix) * 0.8), 6))
    dendrogram(Z, labels=corr_matrix.columns.tolist(), ax=ax, leaf_rotation=90, leaf_font_size=9)
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
        clusters.setdefault(f"cluster_{label}", []).append(dataset)

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
    primary_metric: str,
    episode_params: Optional[str],
    output_dir: str,
    min_models: int,
    include_excluded: bool = False,
    threshold: float = 1.0,
) -> Optional[pd.Series]:
    """Run the full clustering analysis for a single task.

    Args:
        results_dir: Path to the root results directory.
        task_name: The CLI task name.
        primary_metric: The metric to extract and rank by.
        episode_params: Episode params filter (e.g. '1_50').
        output_dir: Directory to write output files.
        min_models: Minimum number of models with complete results.
        include_excluded: If True, include semantic and non-English datasets.
        threshold: Distance threshold for flat clustering.

    Returns:
        A Series of per-model task scores, or None if the task was skipped.
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    df = discover_scores(results_dir, task_name, primary_metric, episode_params, include_excluded)
    if df.empty:
        print(f"  No results found. Skipping.")
        return None

    print(f"  Raw matrix: {len(df)} models × {len(df.columns)} datasets")

    df = df.dropna(axis=0, how="any")
    if len(df) < min_models:
        print(f"  Only {len(df)} models with complete results (need {min_models}). Skipping.")
        return None

    print(f"  Complete matrix: {len(df)} models × {len(df.columns)} datasets")

    if len(df.columns) < 2:
        print(f"  Only {len(df.columns)} dataset(s). Nothing to cluster. Skipping.")
        return None

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
    summary = build_summary(Z, corr, n_models=len(df), threshold=threshold)
    summary_path = os.path.join(output_dir, f"{task_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")
    print(f"  Clusters (threshold={summary['threshold']}):")
    for name, members in summary["clusters"].items():
        print(f"    {name}: {members}")

    # Aggregated scores
    agg = aggregate_scores(df, summary["clusters"])
    agg_path = os.path.join(output_dir, f"{task_name}_aggregated.csv")
    agg.to_csv(agg_path)
    print(f"  Saved aggregated scores: {agg_path}")
    print(f"  Task scores (cluster-aware):")
    for model in agg.index:
        print(f"    {model}: {agg.loc[model, 'task_score']:.4f}")

    return agg["task_score"]


def print_summary_table(
    task_scores: Dict[str, pd.Series],
    task_metrics: Dict[str, str],
    output_dir: str,
) -> None:
    """Print a Markdown table summarizing per-model scores across tasks.

    Bolds the best score in each column. Saves the table to a text file.

    Args:
        task_scores: Mapping from task name to a Series of per-model task scores.
        task_metrics: Mapping from task name to metric name (for column headers).
        output_dir: Directory to save the summary table file.
    """
    if not task_scores:
        return

    columns = {
        f"{task} ({task_metrics[task]})": scores
        for task, scores in task_scores.items()
    }
    df = pd.DataFrame(columns)
    df.index.name = "Model"

    # Bold the best value in each column
    for col in df.columns:
        valid = df[col].dropna()
        if valid.empty:
            df[col] = "—"
            continue
        best_idx = valid.idxmax()
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        df.at[best_idx, col] = f"**{df.at[best_idx, col]}**"

    table = df.to_markdown()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    print(table)
    print()

    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, "summary_table.md")
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f"Saved summary table: {table_path}")


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
    return parser.parse_args()


def main() -> None:
    """Entry point for benchmark clustering analysis."""
    args = parse_args()

    if not args.task and not args.all_tasks:
        print("Error: specify --task <name> or --all-tasks.")
        sys.exit(1)

    tasks = list(TASK_METRICS.keys()) if args.all_tasks else [args.task]
    task_scores: Dict[str, pd.Series] = {}
    effective_metrics: Dict[str, str] = {}

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
        )
        if result is not None:
            task_scores[task] = result

    print_summary_table(task_scores, effective_metrics, args.output_dir)
    print(f"Done. Produced analysis for {len(task_scores)}/{len(tasks)} tasks.")


if __name__ == "__main__":
    main()
