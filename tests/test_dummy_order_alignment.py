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
