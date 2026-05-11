"""Automatic clustering analysis: per-task correlation, dendrogram, summary."""
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from .config import LOW_CONFIDENCE_THRESHOLD, MODEL_CATEGORIES
from .discovery import discover_scores


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
    #   Correlation is between -1, and 1, so dist is between 0, and 2
    dist = np.clip(1 - corr_matrix.values, 0, 2)
    #   Forget about self-correlations (diagonal)
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
    #   Order by clusters found, so that the clusters are visually grouped together
    # in the heatmap
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
    #   For each dataset, assign a cluster label based on the distance threshold
    labels = fcluster(Z, t=threshold, criterion="distance")
    datasets = corr_matrix.columns.tolist()
    #   Create a dictionary mapping cluster labels to lists of datasets
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
    complete_datasets: bool = False,
    allowed_models: Optional[set] = None,
) -> Optional[Tuple[pd.Series, int]]:
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
        complete_datasets: If True, drop datasets with missing models instead
            of dropping models with missing datasets.
        allowed_models: If provided, only include models in this set.

    Returns:
        A tuple of (per-model task scores, number of datasets), or None if
        the task was skipped.
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    df = discover_scores(results_dir, task_name, primary_metric, episode_params, include_excluded, allowed_models)
    if df.empty:
        print(f"  No results found. Skipping.")
        return None

    print(f"  Raw matrix: {len(df)} models × {len(df.columns)} datasets")

    raw_df = df
    if complete_datasets:
        df = df.dropna(axis=1, how="any")
        dropped_datasets = set(raw_df.columns) - set(df.columns)
        if dropped_datasets:
            print(f"  Dropped {len(dropped_datasets)} incomplete dataset(s): {sorted(dropped_datasets)}")
        if len(df) < min_models:
            print(f"  Only {len(df)} models (need {min_models}). Skipping.")
            return None
    else:
        df = df.dropna(axis=0, how="any")
        if len(df) < min_models:
            print(f"  Only {len(df)} models with complete results (need {min_models}). Skipping.")
            dropped = raw_df[raw_df.isna().any(axis=1)]
            if not dropped.empty:
                print(f"  Missing evaluations:")
                for model in dropped.index:
                    missing = [d for d in dropped.columns if pd.isna(dropped.at[model, d])]
                    for dataset in missing:
                        print(f"    ({model}, {dataset})")
            return None

    print(f"  Complete matrix: {len(df)} models × {len(df.columns)} datasets")

    if len(df) < LOW_CONFIDENCE_THRESHOLD:
        print(f"  Warning: fewer than {LOW_CONFIDENCE_THRESHOLD} models — results are low-confidence.")

    os.makedirs(output_dir, exist_ok=True)

    # Save score matrix
    scores_path = os.path.join(output_dir, f"{task_name}_scores.csv")
    df.to_csv(scores_path)
    print(f"  Saved score matrix: {scores_path}")

    if len(df.columns) < 2:
        print(f"  Only {len(df.columns)} dataset(s). Skipping clustering, using raw scores.")
        task_score = df.iloc[:, 0]
        task_score.name = "task_score"
        agg_path = os.path.join(output_dir, f"{task_name}_aggregated.csv")
        task_score.to_csv(agg_path)
        print(f"  Saved aggregated scores: {agg_path}")
        print(f"  Task scores:")
        for model in task_score.index:
            print(f"    {model}: {task_score.loc[model]:.4f}")
        return task_score, len(df.columns)

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

    return agg["task_score"], len(df.columns)


def _get_model_category(
    model: str,
) -> str:
    """Return the category of a model based on MODEL_CATEGORIES config.

    Args:
        model: The model name.

    Returns:
        The category name ("style", "semantic", or "other").
    """
    for category, models in MODEL_CATEGORIES.items():
        if model in models:
            return category
    return "other"


# Colors for each model category in the ranking plot.
_CATEGORY_COLORS = {
    "style": "#2196F3",
    "multilingual": "#90CAF9",
    "semantic": "#FF9800",
    "other": "#9E9E9E",
}

# Multilingual models are style models but rendered in a different shade.
# For grouping purposes they belong to "style".
_MULTILINGUAL_MODELS = {"mstyledistance", "multilingual-style-representation"}

# Display categories (bottom to top in grouped mode)
_DISPLAY_ORDER = ["semantic", "style"]


def _get_display_category(
    model: str,
) -> str:
    """Return the display category for a model (style or semantic only).

    Multilingual models are grouped under style. Models not in any
    category return "other".

    Args:
        model: The model name.

    Returns:
        "style", "semantic", or "other".
    """
    cat = _get_model_category(model)
    if cat == "multilingual":
        return "style"
    return cat


def _get_bar_color(
    model: str,
) -> str:
    """Return the bar color for a model.

    Multilingual models get a distinct shade within the style group.

    Args:
        model: The model name.

    Returns:
        A hex color string.
    """
    if model in _MULTILINGUAL_MODELS:
        return _CATEGORY_COLORS["multilingual"]
    return _CATEGORY_COLORS[_get_display_category(model)]


# Short display names for models with long identifiers.
_DISPLAY_NAMES: Dict[str, str] = {
    "e5-mistral-7b-instruct": "e5-mistral-7b",
    "lisa_checkpoint": "LISA"
}


def _plot_ranking_bars(
    steb_scores: pd.Series,
    ax,
    grouped: bool,
) -> None:
    """Draw horizontal bars on an axis, optionally grouped by category.

    Args:
        steb_scores: Series of STEB scores indexed by model name.
        ax: Matplotlib axis to draw on.
        grouped: If True, group bars by category with gaps between groups.
    """
    from matplotlib.patches import Patch

    x_min = 0.25

    if grouped:
        categories_map = {m: _get_display_category(m) for m in steb_scores.index}

        # Drop "other" models
        steb_scores = steb_scores[[m for m in steb_scores.index if categories_map[m] != "other"]]
        categories_map = {m: c for m, c in categories_map.items() if c != "other"}

        groups = {}
        for cat in _DISPLAY_ORDER:
            members = {m: s for m, s in steb_scores.items() if categories_map[m] == cat}
            if members:
                groups[cat] = pd.Series(members).sort_values(ascending=True)

        y_offset = 0
        y_ticks = []
        y_labels = []

        for cat in _DISPLAY_ORDER:
            if cat not in groups:
                continue
            scores = groups[cat]
            n = len(scores)
            y_pos = np.arange(y_offset, y_offset + n)
            bar_colors = [_get_bar_color(m) for m in scores.index]
            ax.barh(
                y_pos, scores.values,
                color=bar_colors, edgecolor="white", height=0.7,
            )

            # Model names inside the bars, score values at the end
            for i, (model, val) in enumerate(scores.items()):
                label = _DISPLAY_NAMES.get(model, model)
                ax.text(
                    x_min + 0.005, y_pos[i], label,
                    va="center", ha="left", fontsize=14, fontweight="bold",
                    color="white",
                )
                ax.text(val + 0.005, y_pos[i], f"{val:.3f}", va="center", fontsize=14)

            # Vertical line at the best score in this category
            best_val = scores.max()
            ax.axvline(
                x=best_val, color=_CATEGORY_COLORS[cat],
                linestyle="--", linewidth=1.5, alpha=0.6,
            )

            y_ticks.extend(y_pos)
            y_labels.extend([""] * n)
            y_offset += n + 1

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        present = [c for c in _DISPLAY_ORDER if c in groups]
    else:
        sorted_scores = steb_scores.sort_values(ascending=True)
        categories = [_get_display_category(m) for m in sorted_scores.index]
        colors = [_get_bar_color(m) for m in sorted_scores.index]
        n = len(sorted_scores)
        y_pos = np.arange(n)
        ax.barh(y_pos, sorted_scores.values, color=colors, edgecolor="white", height=0.7)
        for i, (model, val) in enumerate(sorted_scores.items()):
            label = _DISPLAY_NAMES.get(model, model)
            ax.text(
                x_min + 0.005, i, label,
                va="center", ha="left", fontsize=14, fontweight="bold",
                color="white",
            )
            ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=14)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([""] * n)
        present = sorted(set(categories))

    ax.set_xlabel("STEB Score", fontsize=18)
    ax.set_title("STEB Score", fontsize=20)
    ax.tick_params(axis="x", labelsize=14)

    legend_handles = [Patch(facecolor=_CATEGORY_COLORS[c], label=c) for c in present]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=14)
    ax.set_xlim(x_min, steb_scores.max() * 1.12)


def plot_model_ranking(
    steb_scores: pd.Series,
    output_dir: str,
) -> List[str]:
    """Generate ranking plots (sorted and grouped) and save them.

    Args:
        steb_scores: Series of STEB scores indexed by model name.
        output_dir: Directory to save the figures.

    Returns:
        List of output file paths [ranking.png, ranking_grouped.png].
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    n = len(steb_scores)

    for grouped, filename in [(False, "ranking.png"), (True, "ranking_grouped.png")]:
        path = os.path.join(output_dir, filename)
        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))
        _plot_ranking_bars(steb_scores, ax, grouped=grouped)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    print(f"  Saved ranking plots: {', '.join(paths)}")
    return paths


def print_summary_table(
    task_scores: Dict[str, pd.Series],
    task_metrics: Dict[str, str],
    task_n_datasets: Dict[str, int],
    output_dir: str,
) -> None:
    """Print a Markdown table summarizing per-model scores across tasks.

    Bolds the best score in each column. Saves the table to a text file.

    Args:
        task_scores: Mapping from task name to a Series of per-model task scores.
        task_metrics: Mapping from task name to metric name (for column headers).
        task_n_datasets: Mapping from task name to number of datasets used.
        output_dir: Directory to save the summary table file.
    """
    if not task_scores:
        return

    col_names = {
        task: f"{task} ({task_metrics[task]})"
        for task in task_scores
    }
    columns = {
        col_names[task]: scores
        for task, scores in task_scores.items()
    }
    df = pd.DataFrame(columns)
    df.index.name = "Model"

    # Build a "# datasets" row
    n_datasets_row = {
        col_names[task]: str(task_n_datasets[task]) if task in task_n_datasets else "—"
        for task in task_scores
    }

    # Bold the best value in each column
    for col in df.columns:
        valid = df[col].dropna()
        if valid.empty:
            df[col] = "—"
            continue
        best_idx = valid.idxmax()
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        df.at[best_idx, col] = f"**{df.at[best_idx, col]}**"

    # Insert the datasets row at the top
    n_datasets_df = pd.DataFrame(n_datasets_row, index=["# datasets"])
    n_datasets_df.index.name = "Model"
    df = pd.concat([n_datasets_df, df])

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
