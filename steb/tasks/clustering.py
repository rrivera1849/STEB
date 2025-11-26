from typing import Dict, Any, List
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import v_measure_score
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

        if embeddings_flat.ndim == 1:
            embeddings_flat = np.array(embeddings_flat).reshape(-1, 1)

        kmeans = MiniBatchKMeans(
            n_clusters=len(set(labels)),
            random_state=42,
            batch_size=32,
            n_init="auto",
        )
        kmeans.fit(embeddings_flat)
        v_measure = v_measure_score(labels, kmeans.labels_)
        return {"v_measure": v_measure}
