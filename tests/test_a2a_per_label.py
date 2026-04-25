"""
Tests for the LABEL-vs-all-other-labels per-label payload emitted by the
all-to-all pair classification task.
"""
import numpy as np

from steb.tasks.all_to_all_pair_classification import AllToAllPairClassificationTask


def _stack_episodes(label_vectors):
    """
    Wraps each label's list of episode-level position-0 embeddings in the
    [[pos0_emb, ...], ...] shape the task expects.

    Args:
        label_vectors: A dict mapping label -> 2D ndarray of shape
            (num_episodes_for_label, dim). Each row is one episode's pos0
            embedding.

    Returns:
        A tuple (embeddings, labels):
          - embeddings: ndarray of shape (total_episodes, 1, dim).
          - labels: list of length total_episodes.
    """
    rows = []
    flat_labels = []
    for label, vectors in label_vectors.items():
        for v in vectors:
            rows.append(v)
            flat_labels.append(label)
    embeddings = np.array(rows)[:, np.newaxis, :]
    return embeddings, flat_labels


def test_per_label_payload_emitted_with_expected_keys():
    """`_per_label` is present and each entry has AUC/EER fields."""
    rng = np.random.default_rng(0)
    label_vectors = {
        "a": rng.normal(loc=[1.0, 0.0], size=(5, 2)),
        "b": rng.normal(loc=[-1.0, 0.0], size=(5, 2)),
        "c": rng.normal(loc=[0.0, 1.0], size=(5, 2)),
    }
    embeddings, labels = _stack_episodes(label_vectors)

    task = AllToAllPairClassificationTask()
    metrics = task.evaluate(embeddings, labels)

    assert "_per_label" in metrics
    per_label = metrics["_per_label"]
    assert set(per_label.keys()) == {"a", "b", "c"}

    for label, scores in per_label.items():
        assert "auc" in scores, f"label {label} missing 'auc'"
        assert "eer" in scores, f"label {label} missing 'eer'"


def test_per_label_label_vs_others_separable_label_scores_high():
    """
    A label whose embeddings are well separated from the rest should get a
    per-label AUC noticeably above chance, while the other labels (which
    overlap) score lower.
    """
    rng = np.random.default_rng(0)
    # "tight" cluster far from the others; the rest overlap around the origin.
    label_vectors = {
        "tight": rng.normal(loc=[10.0, 0.0], scale=0.05, size=(10, 2)),
        "overlap_a": rng.normal(loc=[0.0, 0.0], scale=1.0, size=(10, 2)),
        "overlap_b": rng.normal(loc=[0.0, 0.5], scale=1.0, size=(10, 2)),
    }
    embeddings, labels = _stack_episodes(label_vectors)

    task = AllToAllPairClassificationTask()
    metrics = task.evaluate(embeddings, labels)
    per_label = metrics["_per_label"]

    # The cleanly-separated label's L-vs-others discrimination should be
    # essentially perfect.
    assert per_label["tight"]["auc"] > 0.95
    # The overlapping labels can't be cleanly separated from each other.
    assert per_label["overlap_a"]["auc"] < per_label["tight"]["auc"]
    assert per_label["overlap_b"]["auc"] < per_label["tight"]["auc"]


def test_per_label_global_metrics_unchanged():
    """
    Adding the per-label payload must not alter the global AUC/EER, which
    pre-existed and downstream consumers depend on.
    """
    rng = np.random.default_rng(0)
    label_vectors = {
        "a": rng.normal(loc=[1.0, 0.0], size=(8, 2)),
        "b": rng.normal(loc=[-1.0, 0.0], size=(8, 2)),
    }
    embeddings, labels = _stack_episodes(label_vectors)

    task = AllToAllPairClassificationTask()
    metrics = task.evaluate(embeddings, labels)
    # Global metrics are still at the top level alongside the new internal key.
    for key in ("auc", "eer"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_per_label_skips_singleton_labels():
    """
    A label with only one episode has no within-label pair, so the L-vs-others
    AUC has zero positives and is undefined. The task should silently skip
    that label rather than crash.
    """
    rng = np.random.default_rng(0)
    label_vectors = {
        "many": rng.normal(loc=[1.0, 0.0], size=(5, 2)),
        "singleton": rng.normal(loc=[-1.0, 0.0], size=(1, 2)),
        "other": rng.normal(loc=[0.0, 1.0], size=(5, 2)),
    }
    embeddings, labels = _stack_episodes(label_vectors)

    task = AllToAllPairClassificationTask()
    metrics = task.evaluate(embeddings, labels)
    per_label = metrics["_per_label"]
    assert "singleton" not in per_label
    assert "many" in per_label
    assert "other" in per_label
