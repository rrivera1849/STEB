import numpy as np

from steb.steb_datasets.dummy_order_alignment.loader import (
    load_dummy_order_alignment_dataset,
)
from steb.tasks.order_alignment import OrderAlignmentTask
from steb.validation import validate_config


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


def _build_two_label_inputs():
    """
    Builds a tiny synthetic order_alignment input with two labels.

    Label "good" gets perfectly aligned position embeddings (one-hot per
    position), so its per-label acc_mean should be 1.0. Label "bad" gets
    randomly shuffled embeddings so its acc_mean is below 1.0.

    Returns:
        embeddings: shape (4, 3, 3) numpy array.
        labels: list of length 4, two "good" and two "bad".
    """
    seq_len = 3

    perfect = np.zeros((seq_len, seq_len), dtype=float)
    for pos in range(seq_len):
        perfect[pos, pos] = 1.0

    # "good" pair: identical, perfectly structured.
    good_a = perfect.copy()
    good_b = perfect.copy()

    # "bad" pair: same set of position vectors but reversed in one,
    # forcing alignment to disagree with the expected identity ordering.
    bad_a = perfect.copy()
    bad_b = perfect[::-1].copy()

    embeddings = np.stack([good_a, good_b, bad_a, bad_b], axis=0)
    labels = ["good", "good", "bad", "bad"]
    return embeddings, labels


def test_per_label_payload_emitted():
    """
    OrderAlignmentTask returns a `_per_label` dict keyed by label, with
    per-label acc_mean reflecting only that label's pairs.
    """
    embeddings, labels = _build_two_label_inputs()
    task = OrderAlignmentTask()
    metrics = task.evaluate(embeddings, labels)

    assert "_per_label" in metrics
    per_label = metrics["_per_label"]
    assert set(per_label.keys()) == {"good", "bad"}

    # "good" pair is perfectly aligned: per-label acc_mean must be 1.0.
    assert per_label["good"]["acc_mean"] == 1.0
    # "bad" pair is reversed against itself: per-label acc_mean must be < 1.0.
    assert per_label["bad"]["acc_mean"] < 1.0


def test_per_label_does_not_change_global_means():
    """
    Adding the per-label payload must not alter the global acc_mean
    (which is the pool-then-mean over all pairs across all labels).
    """
    embeddings, labels = _build_two_label_inputs()
    task = OrderAlignmentTask()
    metrics = task.evaluate(embeddings, labels)

    # Global acc_mean is the mean of two per-pair accuracies (one "good"
    # pair gives 1.0, one "bad" pair gives something < 1.0). The exact
    # value depends on the alignment numerics, but it must lie strictly
    # between the two per-label values.
    good = metrics["_per_label"]["good"]["acc_mean"]
    bad = metrics["_per_label"]["bad"]["acc_mean"]
    assert bad < metrics["acc_mean"] < good


def test_validation_rejects_auto_per_label_on_non_order_alignment_task():
    """
    `auto_submetric_per_label` is only valid for `order_alignment`. Setting
    it true on any other task in a config must produce a validation error
    naming the offending task and the flag.
    """
    config = {
        "type": "huggingface",
        "loader_kwargs": {"path": "x", "split": "train"},
        "record_handler": {"text_getter": "text", "label_getter": "label"},
        "tasks": {
            "clustering": {"auto_submetric_per_label": True},
        },
    }
    errors = validate_config(config)
    matching = [e for e in errors if "auto_submetric_per_label" in e and "clustering" in e]
    assert matching, f"expected an error mentioning the flag and the task; got: {errors}"


def test_validation_accepts_auto_per_label_on_order_alignment():
    """A correctly-placed flag passes validation."""
    config = {
        "type": "huggingface",
        "loader_kwargs": {"path": "x", "split": "train"},
        "record_handler": {"text_getter": "text", "label_getter": "label"},
        "tasks": {
            "order_alignment": {"auto_submetric_per_label": True},
        },
    }
    errors = validate_config(config)
    flag_errors = [e for e in errors if "auto_submetric_per_label" in e]
    assert not flag_errors, f"unexpected flag errors: {flag_errors}"


def test_validation_rejects_non_boolean_auto_per_label():
    """The flag must be a boolean, not a string or other type."""
    config = {
        "type": "huggingface",
        "loader_kwargs": {"path": "x", "split": "train"},
        "record_handler": {"text_getter": "text", "label_getter": "label"},
        "tasks": {
            "order_alignment": {"auto_submetric_per_label": "yes"},
        },
    }
    errors = validate_config(config)
    matching = [e for e in errors if "auto_submetric_per_label" in e and "boolean" in e]
    assert matching, f"expected a 'must be a boolean' error; got: {errors}"


