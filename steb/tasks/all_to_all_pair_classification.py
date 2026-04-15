from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..metrics import calculate_pair_classification_metrics
from .base import Task


def _pair_scores_matrix(embeddings_flat: np.ndarray, score_mode: str) -> np.ndarray:
    """Compute pairwise score matrix; higher = more similar."""
    n = embeddings_flat.shape[0]
    if score_mode == "abs_diff":
        # score[i,j] = -|| |e_i - e_j| ||_1
        diff = np.abs(embeddings_flat[:, None, :] - embeddings_flat[None, :, :])
        return -np.sum(diff, axis=2)
    emb = embeddings_flat.reshape(n, -1)
    return cosine_similarity(emb, emb)


class AllToAllPairClassificationTask(Task):
    """
    A task for evaluating pair classification performance (All-to-All).
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Any], score_mode: str = "cosine") -> Dict[str, float]:
        """
        Evaluates the performance of a pair classification model using EER and AUC.

        Always uses the 0th entries (pos0_emb), so the "most" style entries in the processed dataset.

        Args:
            embeddings: The embeddings to evaluate. Expected format: [[pos0_emb, pos1_emb, ...], ...]
            labels: The corresponding labels.
            score_mode: "cosine" (default) or "abs_diff" (for LFTK: -L1 norm of |e1-e2|).

        Returns:
            A dictionary of evaluation metrics, including EER, AUC, and AUC at various FPR thresholds.
        """
        # Extract the 0th position (most style) from record
        embeddings_flat = np.array([episode[0] for episode in embeddings])

        scores = _pair_scores_matrix(embeddings_flat, score_mode)
        y = np.array(labels)
        labels_mat = y.reshape(-1, 1) == y.reshape(1, -1)
        scores = scores[np.triu_indices(scores.shape[0], k=1)]
        labels_flat = labels_mat[np.triu_indices(labels_mat.shape[0], k=1)]

        return calculate_pair_classification_metrics(labels_flat, scores)
