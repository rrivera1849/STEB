"""
Integration tests that run each task type end-to-end on dummy datasets.

These tests use small synthetic datasets and simple embedding models to
verify that the full pipeline (load -> embed -> process -> evaluate) works
for every supported task type.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from steb.core import evaluate, get_model
from steb.dataset_loader import DatasetLoader
from steb.validation import validate_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load a small model once for all integration tests."""
    return get_model("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def output_folder():
    """Provide a temp directory for test outputs, cleaned up after."""
    folder = tempfile.mkdtemp()
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Verify that all dataset configs in the repo are valid."""

    def test_all_configs_valid(self):
        from steb.validation import validate_all_configs

        num_valid, num_invalid = validate_all_configs()
        assert num_invalid == 0, f"{num_invalid} config(s) failed validation"
        assert num_valid > 0

    def test_invalid_config_detected(self):
        errors = validate_config({})
        assert len(errors) > 0
        assert any("type" in e for e in errors)
        assert any("record_handler" in e for e in errors)
        assert any("tasks" in e for e in errors)


# ---------------------------------------------------------------------------
# Clustering task
# ---------------------------------------------------------------------------

class TestClusteringIntegration:
    """End-to-end test for the clustering task."""

    def test_clustering_on_dummy_dataset(self, model, output_folder):
        evaluate(
            model,
            datasets=["dummy_clustering"],
            episode_sizes=[1],
            task_name="clustering",
            n_episodes_per_class=50,
            batch_size=32,
            force_reload=True,
            output_folder=output_folder,
        )

        metrics_path = os.path.join(
            output_folder, "dummy_clustering",
            "all-MiniLM-L6-v2", "1_50", "clustering", "metrics.json",
        )
        assert os.path.exists(metrics_path)

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "v_measure" in metrics
        assert 0.0 <= metrics["v_measure"] <= 1.0


# ---------------------------------------------------------------------------
# All-to-all pair classification task
# ---------------------------------------------------------------------------

class TestAllToAllPairClassificationIntegration:
    """End-to-end test for the all-to-all pair classification task."""

    def test_pair_classification_on_dummy_dataset(self, model, output_folder):
        evaluate(
            model,
            datasets=["dummy_clustering"],
            episode_sizes=[1],
            task_name="all_to_all_pair_classification",
            n_episodes_per_class=50,
            batch_size=32,
            force_reload=True,
            output_folder=output_folder,
        )

        metrics_path = os.path.join(
            output_folder, "dummy_clustering",
            "all-MiniLM-L6-v2", "1_50",
            "all_to_all_pair_classification", "metrics.json",
        )
        assert os.path.exists(metrics_path)

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "auc" in metrics
        assert "eer" in metrics


# ---------------------------------------------------------------------------
# Pre-defined pair classification task
# ---------------------------------------------------------------------------

class TestPreDefinedPairClassificationIntegration:
    """End-to-end test for the pre-defined pair classification task."""

    def test_pre_defined_on_dummy_dataset(self, model, output_folder):
        evaluate(
            model,
            datasets=["dummy_pair_classification"],
            episode_sizes=[1],
            task_name="pre_defined_pair_classification",
            n_episodes_per_class=2,
            batch_size=32,
            force_reload=True,
            output_folder=output_folder,
        )

        metrics_path = os.path.join(
            output_folder, "dummy_pair_classification",
            "all-MiniLM-L6-v2", "1_2",
            "pre_defined_pair_classification", "metrics.json",
        )
        assert os.path.exists(metrics_path)

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "auc" in metrics
        assert "eer" in metrics


# ---------------------------------------------------------------------------
# Order alignment task
# ---------------------------------------------------------------------------

class TestOrderAlignmentIntegration:
    """End-to-end test for the order alignment task."""

    def test_order_alignment_on_dummy_dataset(self, model, output_folder):
        evaluate(
            model,
            datasets=["dummy_order_alignment"],
            episode_sizes=[1],
            task_name="order_alignment",
            n_episodes_per_class=2,
            batch_size=32,
            force_reload=True,
            output_folder=output_folder,
        )

        metrics_path = os.path.join(
            output_folder, "dummy_order_alignment",
            "all-MiniLM-L6-v2", "1_2",
            "order_alignment", "metrics.json",
        )
        assert os.path.exists(metrics_path)

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "acc_mean" in metrics
        assert "distractor_acc_mean" in metrics


# ---------------------------------------------------------------------------
# Probing task
# ---------------------------------------------------------------------------

class TestProbingIntegration:
    """End-to-end test for the probing task."""

    def test_probing_on_dummy_dataset(self, model, output_folder):
        evaluate(
            model,
            datasets=["dummy_probing"],
            episode_sizes=[1],
            task_name="probing",
            n_episodes_per_class=1,
            batch_size=32,
            force_reload=True,
            output_folder=output_folder,
        )

        metrics_path = os.path.join(
            output_folder, "dummy_probing",
            "all-MiniLM-L6-v2", "1_1",
            "probing", "metrics.json",
        )
        assert os.path.exists(metrics_path)

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "task_0" in metrics
        assert "average" in metrics


# ---------------------------------------------------------------------------
# Cache key includes seed
# ---------------------------------------------------------------------------

class TestCacheKeyIncludesSeed:
    """Verify that different seeds produce different cache paths."""

    def test_different_seeds_different_paths(self):
        loader_a = DatasetLoader("dummy_clustering", seed=42)
        loader_b = DatasetLoader("dummy_clustering", seed=99)
        assert loader_a._get_dataset_path() != loader_b._get_dataset_path()
        assert "seed42" in loader_a._get_dataset_path()
        assert "seed99" in loader_b._get_dataset_path()
