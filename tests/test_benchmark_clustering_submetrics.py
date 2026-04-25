"""
Tests for the submetric-aware extensions to scripts/benchmark_clustering.py:
- ClusterEntry parsing from YAML.
- _read_submetric_scores walking a results tree.
- build_manual_cluster_tables aggregating across plain and submetric entries.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Tests live under tests/, scripts/ is a sibling — extend sys.path explicitly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_clustering as bc  # noqa: E402


def _write_metrics(
    results_dir: Path,
    dataset: str,
    model: str,
    ep: str,
    task: str,
    metrics: dict,
) -> None:
    """Drop a metrics.json into results/<dataset>/<model>/<ep>/<task>/."""
    path = results_dir / dataset / model / ep / task
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f)


def _write_clusters_yaml(path: Path, clusters: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(clusters, f)


# ---------------------------------------------------------------------------
# load_manual_clusters: YAML parsing
# ---------------------------------------------------------------------------

def test_load_manual_clusters_string_entries_unchanged(tmp_path: Path):
    """Plain string entries still produce ClusterEntry(dataset=...) only."""
    clusters_path = tmp_path / "clusters.yaml"
    _write_clusters_yaml(clusters_path, {
        "topic": {"description": "x", "datasets": ["ds_a", "ds_b"]},
    })
    out = bc.load_manual_clusters(str(clusters_path))
    assert list(out.keys()) == ["topic"]
    entries = out["topic"]
    assert all(e.submetrics is None for e in entries)
    assert [e.dataset for e in entries] == ["ds_a", "ds_b"]


def test_load_manual_clusters_submetric_entries(tmp_path: Path):
    """Mapping entries with submetrics parse into ClusterEntry with task+submetrics."""
    clusters_path = tmp_path / "clusters.yaml"
    _write_clusters_yaml(clusters_path, {
        "style_similarity": {
            "description": "...",
            "datasets": [
                {"dataset": "STEL", "task": "order_alignment",
                 "submetrics": ["formal", "complex"]},
                "plain_dataset",  # mixed string + dict entries are allowed
            ],
        },
    })
    out = bc.load_manual_clusters(str(clusters_path))
    sub_entry, plain_entry = out["style_similarity"]
    assert sub_entry.dataset == "STEL"
    assert sub_entry.task == "order_alignment"
    assert sub_entry.submetrics == ("formal", "complex")
    assert plain_entry.dataset == "plain_dataset"
    assert plain_entry.submetrics is None


def test_load_manual_clusters_submetric_without_task_errors(tmp_path: Path):
    """`submetrics:` requires `task:` — otherwise a clear ValueError."""
    clusters_path = tmp_path / "clusters.yaml"
    _write_clusters_yaml(clusters_path, {
        "bad": {"description": "x", "datasets": [
            {"dataset": "STEL", "submetrics": ["formal"]},
        ]},
    })
    with pytest.raises(ValueError, match="task"):
        bc.load_manual_clusters(str(clusters_path))


def test_cluster_entry_label():
    """Plain entries use the dataset name; submetric entries embed task+subs."""
    plain = bc.ClusterEntry(dataset="STEL")
    sub = bc.ClusterEntry(
        dataset="STEL", task="order_alignment",
        submetrics=("formal", "complex"),
    )
    assert plain.label == "STEL"
    assert sub.label == "STEL[formal+complex]@order_alignment"


# ---------------------------------------------------------------------------
# _read_submetric_scores: walks the results tree
# ---------------------------------------------------------------------------

def test_read_submetric_scores_pulls_named_submetric_per_model(tmp_path: Path):
    """One submetric across two models -> {model: score} for both."""
    results = tmp_path / "results"
    metrics_a = {
        "acc_mean": 0.5, "distractor_acc_mean": 0.4,
        "submetrics": {
            "formal": {"acc_mean": 0.7, "distractor_acc_mean": 0.6},
            "complex": {"acc_mean": 0.55, "distractor_acc_mean": 0.5},
        },
    }
    metrics_b = {
        "acc_mean": 0.5, "distractor_acc_mean": 0.4,
        "submetrics": {
            "formal": {"acc_mean": 0.3, "distractor_acc_mean": 0.25},
            "complex": {"acc_mean": 0.45, "distractor_acc_mean": 0.4},
        },
    }
    _write_metrics(results, "STEL", "model_a", "1_50", "order_alignment", metrics_a)
    _write_metrics(results, "STEL", "model_b", "1_50", "order_alignment", metrics_b)

    scores = bc._read_submetric_scores(
        str(results), "STEL", "order_alignment", "formal",
        primary_metric="acc_mean",
        episode_params="1_50",
    )
    assert scores == {"model_a": 0.7, "model_b": 0.3}


def test_read_submetric_scores_skips_models_without_the_submetric(tmp_path: Path):
    """A model whose metrics.json lacks the submetric is silently skipped."""
    results = tmp_path / "results"
    _write_metrics(results, "STEL", "model_with", "1_50", "order_alignment", {
        "acc_mean": 0.5,
        "submetrics": {"formal": {"acc_mean": 0.7}},
    })
    _write_metrics(results, "STEL", "model_without", "1_50", "order_alignment", {
        "acc_mean": 0.5,
        # No submetrics block at all.
    })

    scores = bc._read_submetric_scores(
        str(results), "STEL", "order_alignment", "formal",
        primary_metric="acc_mean",
        episode_params="1_50",
    )
    assert scores == {"model_with": 0.7}


# ---------------------------------------------------------------------------
# build_manual_cluster_tables: end-to-end aggregation
# ---------------------------------------------------------------------------

def _setup_minimal_results(tmp_path: Path) -> Path:
    """
    Minimal results layout exercising both plain and submetric entries.

    STEL/model_a: top-level acc_mean=0.5, formal=0.8, complex=0.6
    STEL/model_b: top-level acc_mean=0.5, formal=0.4, complex=0.2

    Plus a plain dataset 'other_ds' with order_alignment scores so we can
    confirm the plain-entry path still works.
    """
    results = tmp_path / "results"
    _write_metrics(results, "STEL", "model_a", "1_50", "order_alignment", {
        "acc_mean": 0.5, "distractor_acc_mean": 0.4,
        "submetrics": {
            "formal": {"acc_mean": 0.8, "distractor_acc_mean": 0.7},
            "complex": {"acc_mean": 0.6, "distractor_acc_mean": 0.55},
        },
    })
    _write_metrics(results, "STEL", "model_b", "1_50", "order_alignment", {
        "acc_mean": 0.5, "distractor_acc_mean": 0.4,
        "submetrics": {
            "formal": {"acc_mean": 0.4, "distractor_acc_mean": 0.3},
            "complex": {"acc_mean": 0.2, "distractor_acc_mean": 0.15},
        },
    })
    return results


def test_build_manual_cluster_tables_submetric_entry_averages_within_dataset(tmp_path: Path):
    """
    A submetric entry contributes mean(formal, complex) per model. With a
    single submetric entry, that mean IS the cluster cell value.
    """
    results = _setup_minimal_results(tmp_path)
    clusters = {
        "style_similarity": [
            bc.ClusterEntry(
                dataset="STEL", task="order_alignment",
                submetrics=("formal", "complex"),
            ),
        ],
    }
    tables, column_datasets = bc.build_manual_cluster_tables(
        str(results), clusters, episode_params="1_50",
    )
    assert "style_similarity" in tables
    df = tables["style_similarity"]
    # column header uses the task's primary metric per TASK_METRICS
    col = "order_alignment (distractor_acc_mean)"
    assert col in df.columns
    # mean(0.7, 0.55) for model_a; mean(0.3, 0.15) for model_b
    assert df.loc["model_a", col] == pytest.approx((0.7 + 0.55) / 2)
    assert df.loc["model_b", col] == pytest.approx((0.3 + 0.15) / 2)
    # Column-dataset listing carries the entry label
    assert column_datasets["style_similarity"][col] == \
        ["STEL[formal+complex]@order_alignment"]
