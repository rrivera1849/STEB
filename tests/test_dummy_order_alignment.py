import itertools

import numpy as np
import pytest

from steb.steb_datasets.dummy_order_alignment.loader import (
    load_dummy_order_alignment_dataset,
)
from steb.tasks.order_alignment import OrderAlignmentTask


def test_dummy_order_alignment_dataset_has_multiple_sequences_per_label():
    records = load_dummy_order_alignment_dataset(_path="unused")
    assert records, "Dummy order-alignment dataset should not be empty"

    by_label = {}
    for rec in records:
        by_label.setdefault(rec["label"], []).append(rec["text"])

    for label, sequences in by_label.items():
        # At least two text lists per style label
        assert len(sequences) >= 2, f"Label '{label}' should have at least two sequences"
        # For now: all sequences for a label should share the same length
        lengths = {len(seq) for seq in sequences}
        assert len(lengths) == 1, f"Sequences for label '{label}' must share the same length"


def test_order_alignment_on_dummy_dataset_yields_high_accuracy():
    records = load_dummy_order_alignment_dataset(_path="unused")
    assert len(records) >= 2

    seq_len = len(records[0]["text"])
    assert seq_len >= 2

    num_lists = len(records)
    # Build position-based embeddings so that same positions across lists are identical.
    #   i.e., [[1, 0, 0], [0, 1, 0], [0, 0, 1]] for seq_len=3
    embeddings = np.zeros((num_lists, seq_len, seq_len), dtype=float)
    for i in range(num_lists):
        for pos in range(seq_len):
            embeddings[i, pos, pos] = 1.0

    labels = [rec["label"] for rec in records]

    task = OrderAlignmentTask()
    metrics = task.evaluate(embeddings, labels)

    # Perfect alignment expected for perfectly structured embeddings
    assert metrics["acc_mean"] == 1.0
    assert metrics["distractor_acc_mean"] == 1.0


def _multi_label_synthetic_input(seed: int = 0):
    """Build a small multi-label synthetic input for OrderAlignmentTask.

    Returns (embeddings, labels) with two labels and an unequal number of
    text lists per label, so the per-label pair counts differ. This lets
    the weighted-average check actually catch a bug where _per_label is
    computed with the wrong subset of pairs.
    """
    rng = np.random.default_rng(seed)
    # 3 lists for label 'a' (3 within-label pairs) and 2 for label 'b' (1 pair)
    labels = ["a", "a", "a", "b", "b"]
    embeddings = rng.standard_normal((len(labels), 4, 8))
    return embeddings, labels


def test_per_label_keys_are_present_and_match_labels():
    embeddings, labels = _multi_label_synthetic_input()

    metrics = OrderAlignmentTask().evaluate(embeddings, labels)

    assert "_per_label" in metrics, "evaluate() must include a _per_label dict"
    per_label = metrics["_per_label"]
    assert isinstance(per_label, dict)
    # Keys are the labels (cast to str for JSON safety)
    assert set(per_label.keys()) == {str(l) for l in set(labels)}
    # Each per-label entry exposes the same two metric keys as top-level
    for label_metrics in per_label.values():
        assert set(label_metrics.keys()) == {"acc_mean", "distractor_acc_mean"}
        assert all(isinstance(v, float) for v in label_metrics.values())


def test_per_label_values_aggregate_to_top_level_means():
    """Weighted mean of per-label values must equal the top-level means.

    Weights are the number of within-label pair comparisons each label
    contributes — for n items per label, that's C(n, 2) baseline pairs and
    2 * C(n, 2) distractor accuracies (two distractor variants per pair).
    Since both top-level and per-label means use the same per-pair weights,
    the per-label-weighted mean should reproduce the top-level mean.
    """
    embeddings, labels = _multi_label_synthetic_input()

    metrics = OrderAlignmentTask().evaluate(embeddings, labels)
    per_label = metrics["_per_label"]

    counts = {label: labels.count(label) for label in set(labels)}
    pair_counts = {label: len(list(itertools.combinations(range(n), 2)))
                   for label, n in counts.items()}

    total_pairs = sum(pair_counts.values())
    expected_acc_mean = sum(
        per_label[str(label)]["acc_mean"] * pair_counts[label]
        for label in pair_counts
    ) / total_pairs
    expected_distractor_acc_mean = sum(
        per_label[str(label)]["distractor_acc_mean"] * pair_counts[label]
        for label in pair_counts
    ) / total_pairs

    assert metrics["acc_mean"] == pytest.approx(expected_acc_mean)
    assert metrics["distractor_acc_mean"] == pytest.approx(expected_distractor_acc_mean)
