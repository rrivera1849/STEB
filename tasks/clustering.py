from typing import Dict, Any, List
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import v_measure_score
from tasks.base import Task

class ClusteringTask(Task):
    def evaluate(self, embeddings: np.ndarray, labels: List[Any]) -> Dict[str, float]:
        """
        Train a K-Means model and evaluate its performance using V-measure.
        """
        if embeddings.ndim == 1:
            embeddings = np.array(embeddings).reshape(-1, 1)

        kmeans = MiniBatchKMeans(
            n_clusters=len(set(labels)),
            random_state=42,
            batch_size=32,
            n_init="auto",
        )
        kmeans.fit(embeddings)
        v_measure = v_measure_score(labels, kmeans.labels_)
        return {"v_measure": v_measure}
