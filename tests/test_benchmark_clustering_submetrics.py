"""
Tests for the flag-aware extensions to scripts/benchmark_clustering.py.

Covers:
- _parse_entry on each form (plain, --task only, --submetrics only, both,
  unknown flag).
- _read_submetric_scores walking a results tree.
- _infer_task picking a unique task and erroring on ambiguity / no match.
- build_manual_cluster_tables: backward-compat regression for plain entries,
  end-to-end submetric averaging, mixed plain + flagged entries.
"""
import json
import sys
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import pytest
import yaml

# Tests live under tests/, scripts/ is a sibling — extend sys.path explicitly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_clustering as bc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

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


def _patch_supported_datasets(supported: dict):
    """
    Patch ``get_supported_datasets`` inside benchmark_clustering's module
    namespace so the discovery helpers see the fixture's tasks/datasets.

    Args:
        supported: ``{task_name: [dataset_names]}`` mapping that the patched
            function returns.
    """
    def fake(task_name: str):
        return supported.get(task_name, [])
    return patch.object(bc, "get_supported_datasets", side_effect=fake)


# ---------------------------------------------------------------------------
# _parse_entry
# ---------------------------------------------------------------------------

def test_parse_entry_plain():
    """A bare dataset string is a plain entry — no task, no submetrics."""
    entry = bc._parse_entry("STEL")
    assert entry == bc.ClusterEntry(dataset="STEL", task=None, submetrics=None)
    assert entry.label == "STEL"


def test_parse_entry_task_only():
    entry = bc._parse_entry("STEL --task order_alignment")
    assert entry.dataset == "STEL"
    assert entry.task == "order_alignment"
    assert entry.submetrics is None
    assert entry.label == "STEL@order_alignment"


def test_parse_entry_submetrics_only():
    entry = bc._parse_entry("STEL --submetrics formal complex")
    assert entry.dataset == "STEL"
    assert entry.task is None
    assert entry.submetrics == ("formal", "complex")
    # No task in the label since none was given.
    assert "formal+complex" in entry.label


def test_parse_entry_both_flags_in_either_order():
    a = bc._parse_entry("STEL --task order_alignment --submetrics formal complex")
    b = bc._parse_entry("STEL --submetrics formal complex --task order_alignment")
    assert a == b
    assert a.label == "STEL[formal+complex]@order_alignment"


def test_parse_entry_unknown_flag_raises():
    with pytest.raises(ValueError, match="Could not parse"):
        bc._parse_entry("STEL --bogus value")


def test_parse_entry_empty_string_raises():
    with pytest.raises(ValueError, match="Empty"):
        bc._parse_entry("")


# ---------------------------------------------------------------------------
# load_manual_clusters
# ---------------------------------------------------------------------------

def test_load_manual_clusters_plain_strings_unchanged(tmp_path: Path):
    """All-plain YAML still produces ClusterEntry(dataset=...) entries only."""
    clusters_path = tmp_path / "clusters.yaml"
    _write_clusters_yaml(clusters_path, {
        "topic": {"description": "x", "datasets": ["ds_a", "ds_b"]},
    })
    out = bc.load_manual_clusters(str(clusters_path))
    assert [e.dataset for e in out["topic"]] == ["ds_a", "ds_b"]
    assert all(e.task is None and e.submetrics is None for e in out["topic"])


def test_load_manual_clusters_mixed_entries(tmp_path: Path):
    """Plain and flagged strings can mix in one cluster."""
    clusters_path = tmp_path / "clusters.yaml"
    _write_clusters_yaml(clusters_path, {
        "style_similarity": {
            "description": "...",
            "datasets": [
                "plain_dataset",
                "STEL --submetrics formal complex",
                "CoDS --task order_alignment",
            ],
        },
    })
    plain, sub, scoped = bc.load_manual_clusters(str(clusters_path))["style_similarity"]
    assert plain.dataset == "plain_dataset" and plain.task is None
    assert sub.dataset == "STEL" and sub.submetrics == ("formal", "complex")
    assert scoped.dataset == "CoDS" and scoped.task == "order_alignment"


def test_load_manual_clusters_rejects_non_string_entries(tmp_path: Path):
    """Mappings or other non-string entries get a clear error."""
    clusters_path = tmp_path / "clusters.yaml"
    _write_clusters_yaml(clusters_path, {
        "bad": {"description": "x", "datasets": [{"dataset": "STEL"}]},
    })
    with pytest.raises(ValueError, match="non-string entry"):
        bc.load_manual_clusters(str(clusters_path))


# ---------------------------------------------------------------------------
# _read_submetric_scores
# ---------------------------------------------------------------------------

def test_read_submetric_scores_pulls_named_submetric_per_model(tmp_path: Path):
    """Two models: each contributes their submetric's primary metric value."""
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
    """Models whose metrics.json lacks the submetric are silently skipped."""
    results = tmp_path / "results"
    _write_metrics(results, "STEL", "with", "1_50", "order_alignment", {
        "acc_mean": 0.5, "submetrics": {"formal": {"acc_mean": 0.7}},
    })
    _write_metrics(results, "STEL", "without", "1_50", "order_alignment", {
        "acc_mean": 0.5,  # no submetrics block
    })
    scores = bc._read_submetric_scores(
        str(results), "STEL", "order_alignment", "formal",
        primary_metric="acc_mean",
        episode_params="1_50",
    )
    assert scores == {"with": 0.7}


# ---------------------------------------------------------------------------
# _infer_task
# ---------------------------------------------------------------------------

def test_infer_task_unique_match(tmp_path: Path):
    """If only one task has the named submetric, that task is returned."""
    results = tmp_path / "results"
    _write_metrics(results, "STEL", "m", "1_50", "order_alignment", {
        "acc_mean": 0.5,
        "submetrics": {"formal": {"acc_mean": 0.5}, "complex": {"acc_mean": 0.5}},
    })
    with _patch_supported_datasets({
        "order_alignment": ["STEL"],
        "clustering": [],
    }):
        inferred = bc._infer_task(str(results), "STEL", ("formal",), episode_params="1_50")
    assert inferred == "order_alignment"


def test_infer_task_no_match_raises(tmp_path: Path):
    results = tmp_path / "results"
    _write_metrics(results, "STEL", "m", "1_50", "order_alignment", {
        "acc_mean": 0.5,
        "submetrics": {"different_label": {"acc_mean": 0.5}},
    })
    with _patch_supported_datasets({"order_alignment": ["STEL"]}):
        with pytest.raises(ValueError, match="no task's metrics.json contains"):
            bc._infer_task(str(results), "STEL", ("formal",), episode_params="1_50")


def test_infer_task_multiple_match_raises(tmp_path: Path):
    """Same submetric label under two tasks → ambiguous, ask for --task."""
    results = tmp_path / "results"
    _write_metrics(results, "CoDS", "m", "1_50", "order_alignment", {
        "acc_mean": 0.5,
        "submetrics": {"shared_label": {"acc_mean": 0.5}},
    })
    _write_metrics(results, "CoDS", "m", "1_50", "clustering", {
        "v_measure": 0.5,
        "submetrics": {"shared_label": {"v_measure": 0.5}},
    })
    with _patch_supported_datasets({
        "order_alignment": ["CoDS"],
        "clustering": ["CoDS"],
    }):
        with pytest.raises(ValueError, match="Ambiguous task"):
            bc._infer_task(str(results), "CoDS", ("shared_label",), episode_params="1_50")


# ---------------------------------------------------------------------------
# build_manual_cluster_tables
# ---------------------------------------------------------------------------

def _setup_results(tmp_path: Path) -> Path:
    """Three datasets, two models, both clustering and order_alignment.

    STEL/SynthSTEL: order_alignment with formal/complex submetrics.
    other_topic_ds: clustering only (used as a "plain" entry control).
    """
    results = tmp_path / "results"

    # STEL: model_a high, model_b low on formal+complex
    _write_metrics(results, "STEL", "model_a", "1_50", "order_alignment", {
        "acc_mean": 0.6, "distractor_acc_mean": 0.55,
        "submetrics": {
            "formal":  {"acc_mean": 0.8, "distractor_acc_mean": 0.7},
            "complex": {"acc_mean": 0.6, "distractor_acc_mean": 0.55},
        },
    })
    _write_metrics(results, "STEL", "model_b", "1_50", "order_alignment", {
        "acc_mean": 0.4, "distractor_acc_mean": 0.3,
        "submetrics": {
            "formal":  {"acc_mean": 0.4, "distractor_acc_mean": 0.3},
            "complex": {"acc_mean": 0.2, "distractor_acc_mean": 0.15},
        },
    })

    # SynthSTEL: similar shape, different numbers.
    _write_metrics(results, "SynthSTEL", "model_a", "1_50", "order_alignment", {
        "acc_mean": 0.5, "distractor_acc_mean": 0.45,
        "submetrics": {
            "formal":  {"acc_mean": 0.65, "distractor_acc_mean": 0.6},
            "complex": {"acc_mean": 0.55, "distractor_acc_mean": 0.5},
        },
    })
    _write_metrics(results, "SynthSTEL", "model_b", "1_50", "order_alignment", {
        "acc_mean": 0.5, "distractor_acc_mean": 0.45,
        "submetrics": {
            "formal":  {"acc_mean": 0.3, "distractor_acc_mean": 0.25},
            "complex": {"acc_mean": 0.5, "distractor_acc_mean": 0.45},
        },
    })

    # Plain control: a clustering dataset with no submetrics.
    _write_metrics(results, "other_topic_ds", "model_a", "1_50", "clustering", {"v_measure": 0.7})
    _write_metrics(results, "other_topic_ds", "model_b", "1_50", "clustering", {"v_measure": 0.3})

    return results


def _supported_for_results():
    """Match the fixture's task↔dataset support."""
    return {
        "order_alignment": ["STEL", "SynthSTEL"],
        "clustering": ["other_topic_ds"],
        "all_to_all_pair_classification": [],
        "pre_defined_pair_classification": [],
        "retrieval": [],
        "probing": [],
    }


def test_build_tables_plain_entries_unchanged(tmp_path: Path):
    """
    Backward-compat regression: a cluster of plain entries produces exactly
    the (model, task) → score values you'd compute by hand. No off-by-one,
    no new columns, no unexpected rows.
    """
    results = _setup_results(tmp_path)
    clusters = {
        "topic_only": [bc.ClusterEntry(dataset="other_topic_ds")],
    }
    with _patch_supported_datasets(_supported_for_results()):
        tables, column_datasets = bc.build_manual_cluster_tables(
            str(results), clusters, episode_params="1_50",
        )
    df = tables["topic_only"]
    col = "clustering (v_measure)"
    assert list(df.columns) == [col]
    assert df.loc["model_a", col] == pytest.approx(0.7)
    assert df.loc["model_b", col] == pytest.approx(0.3)
    # Column listing uses the bare dataset name (label == dataset for plain entries).
    assert column_datasets["topic_only"][col] == ["other_topic_ds"]


def test_build_tables_submetric_entry_averages_within_dataset(tmp_path: Path):
    """A `--submetrics` entry contributes mean(formal, complex) per model."""
    results = _setup_results(tmp_path)
    clusters = {
        "style_similarity": [
            bc.ClusterEntry(
                dataset="STEL", task="order_alignment",
                submetrics=("formal", "complex"),
            ),
        ],
    }
    with _patch_supported_datasets(_supported_for_results()):
        tables, column_datasets = bc.build_manual_cluster_tables(
            str(results), clusters, episode_params="1_50",
        )
    col = "order_alignment (distractor_acc_mean)"
    df = tables["style_similarity"]
    # mean(0.7, 0.55) for model_a; mean(0.3, 0.15) for model_b
    assert df.loc["model_a", col] == pytest.approx((0.7 + 0.55) / 2)
    assert df.loc["model_b", col] == pytest.approx((0.3 + 0.15) / 2)
    assert column_datasets["style_similarity"][col] == [
        "STEL[formal+complex]@order_alignment",
    ]


def test_build_tables_mixed_entries_rows_dont_collide(tmp_path: Path):
    """
    A plain STEL entry and a STEL submetric entry coexist in the same cluster.
    Plain row uses the bare dataset name; flagged row uses the entry label.
    """
    results = _setup_results(tmp_path)
    clusters = {
        "mixed": [
            bc.ClusterEntry(dataset="STEL"),  # plain
            bc.ClusterEntry(
                dataset="STEL", task="order_alignment",
                submetrics=("formal",),
            ),
        ],
    }
    with _patch_supported_datasets(_supported_for_results()):
        tables, column_datasets = bc.build_manual_cluster_tables(
            str(results), clusters, episode_params="1_50",
        )
    col = "order_alignment (distractor_acc_mean)"
    labels = column_datasets["mixed"][col]
    assert "STEL" in labels
    assert "STEL[formal]@order_alignment" in labels
    # Cluster cell is the mean of (plain row, flagged row) per model.
    df = tables["mixed"]
    expected_a = (0.55 + 0.7) / 2  # plain top-level + formal-only
    expected_b = (0.30 + 0.3) / 2
    assert df.loc["model_a", col] == pytest.approx(expected_a)
    assert df.loc["model_b", col] == pytest.approx(expected_b)


def test_build_tables_submetric_inferred_task(tmp_path: Path):
    """`--submetrics` without `--task` resolves to a unique task automatically."""
    results = _setup_results(tmp_path)
    clusters = {
        "style_similarity": [
            bc.ClusterEntry(
                dataset="STEL", task=None,
                submetrics=("formal", "complex"),
            ),
        ],
    }
    with _patch_supported_datasets(_supported_for_results()):
        tables, column_datasets = bc.build_manual_cluster_tables(
            str(results), clusters, episode_params="1_50",
        )
    col = "order_alignment (distractor_acc_mean)"
    df = tables["style_similarity"]
    assert df.loc["model_a", col] == pytest.approx((0.7 + 0.55) / 2)
