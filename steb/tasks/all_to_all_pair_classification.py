from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..metrics import calculate_pair_classification_metrics
from .base import Task


class AllToAllPairClassificationTask(Task):
    """
    A task for evaluating pair classification performance (All-to-All).
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Any]) -> Dict[str, float]:
        """
        Evaluates the performance of a pair classification model using EER and AUC.

        Always uses the 0th entries (pos0_emb), so the "most" style entries in the processed dataset.

        Args:
            embeddings: The embeddings to evaluate. Expected format: [[pos0_emb, pos1_emb, ...], ...]
            labels: The corresponding labels.

        Returns:
            A dictionary of evaluation metrics, including EER, AUC, and AUC at various FPR thresholds.
        """
        # Extract the 0th position (most style) from record
        embeddings_flat = np.array([episode[0] for episode in embeddings])

        scores = cosine_similarity(
            embeddings_flat.reshape(-1, embeddings_flat[0].shape[-1]),
            embeddings_flat.reshape(-1, embeddings_flat[0].shape[-1])
        )
        y = np.array(labels)
        labels = y.reshape(-1, 1) == y.reshape(1, -1)
        scores = scores[np.triu_indices(scores.shape[0], k=1)]
        labels = labels[np.triu_indices(labels.shape[0], k=1)]

        return calculate_pair_classification_metrics(labels, scores)
