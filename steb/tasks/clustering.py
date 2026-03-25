from typing import Any, Dict, List

import numpy as np

from ..metrics import calculate_clustering_metrics
from .base import Task


class ClusteringTask(Task):
    """
    A task for evaluating clustering performance.
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Any]) -> Dict[str, float]:
        """
        Trains a K-Means model and evaluates its performance using V-measure.

        Always uses the 0th entries (pos0_emb), so the "most" style entries in the processed dataset.

        Args:
            embeddings: The embeddings to evaluate. Expected format: [[pos0_emb, pos1_emb, ...], ...]
            labels: The corresponding labels.

        Returns:
            A dictionary containing the V-measure score.
        """
        # Extract the 0th position (most style) from each record
        embeddings_flat = np.array([episode[0] for episode in embeddings])

        return calculate_clustering_metrics(embeddings_flat, labels)
