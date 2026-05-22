from typing import Any, Dict, List

import numpy as np

from ..metrics import calculate_clustering_metrics
from .base import Task


class ClusteringTask(Task):
    """
    A task for evaluating clustering performance.
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Any], distance_mode: str = "euclidean") -> Dict[str, float]:
        """
        Clusters embeddings and evaluates using V-measure.

        Always uses the 0th entries (pos0_emb), so the "most" style entries in the processed dataset.

        Args:
            embeddings: The embeddings to evaluate. Expected format: [[pos0_emb, pos1_emb, ...], ...]
            labels: The corresponding labels.
            distance_mode: "euclidean" (K-Means) or "l1_diff" (for LFTK: L1 norm of |e_i-e_j|).

        Returns:
            A dictionary containing the V-measure score.
        """
        embeddings_flat = np.array([episode[0] for episode in embeddings])

        return calculate_clustering_metrics(embeddings_flat, labels, distance_mode=distance_mode)
