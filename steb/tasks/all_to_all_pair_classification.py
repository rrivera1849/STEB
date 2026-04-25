from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..metrics import calculate_pair_classification_metrics
from .base import Task


class AllToAllPairClassificationTask(Task):
    """
    A task for evaluating pair classification performance (All-to-All).
    """
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: List[Any],
    ) -> Dict[str, Any]:
        """
        Evaluates the performance of a pair classification model using EER and AUC.

        Always uses the 0th entries (pos0_emb), so the "most" style entries in the processed dataset.

        In addition to the global same-label-vs-different-label metrics, emits a
        per-label "LABEL vs. all other labels" payload under the internal key
        ``_per_label``. For each unique label L, the per-label metric restricts
        to pairs where at least one endpoint has label L; among those pairs, the
        positive class is "both endpoints are L" and the negative class is
        "exactly one endpoint is L". This answers, for each L, "given a pair
        involving L, can the model tell whether the other side is also L?".
        ``core.py`` strips the internal key before serialisation and lifts it
        into ``metrics["submetrics"]`` for tasks listed in ``AUTO_PER_LABEL_TASKS``.

        Args:
            embeddings: The embeddings to evaluate. Expected format: [[pos0_emb, pos1_emb, ...], ...]
            labels: The corresponding labels.

        Returns:
            A dictionary of evaluation metrics, including EER, AUC, and AUC at
            various FPR thresholds, plus the internal ``_per_label`` payload.
        """
        embeddings_flat = np.array([episode[0] for episode in embeddings])

        sim_matrix = cosine_similarity(
            embeddings_flat.reshape(-1, embeddings_flat[0].shape[-1]),
            embeddings_flat.reshape(-1, embeddings_flat[0].shape[-1])
        )
        y = np.array(labels)
        label_eq_matrix = y.reshape(-1, 1) == y.reshape(1, -1)
        triu_idx = np.triu_indices(sim_matrix.shape[0], k=1)
        scores = sim_matrix[triu_idx]
        pair_same_label = label_eq_matrix[triu_idx]

        result: Dict[str, Any] = calculate_pair_classification_metrics(pair_same_label, scores)

        # Per-label "LABEL vs all others": for each unique label L, restrict to
        # pairs where at least one endpoint is L, then compute AUC/EER with
        # positive=both-L vs negative=exactly-one-L. Reuses the precomputed
        # similarity scores; the per-label cost is just K mask-and-metric calls.
        y_i = y[triu_idx[0]]
        y_j = y[triu_idx[1]]
        per_label: Dict[str, Dict[str, float]] = {}
        for label_value in np.unique(y):
            touches = (y_i == label_value) | (y_j == label_value)
            if not touches.any():
                continue
            both = (y_i == label_value) & (y_j == label_value)
            sub_scores = scores[touches]
            sub_labels = both[touches]
            # Need at least one positive and one negative for AUC/EER to be
            # well-defined. A label with zero "both-L" pairs (singleton class)
            # or zero "L vs other" pairs (everything is L) is skipped.
            if sub_labels.any() and not sub_labels.all():
                per_label[str(label_value)] = calculate_pair_classification_metrics(
                    sub_labels, sub_scores,
                )

        result["_per_label"] = per_label
        return result
