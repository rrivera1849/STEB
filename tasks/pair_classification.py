from typing import Dict, Any, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tasks.base import Task

def calculate_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

class PairClassificationTask(Task):
    def evaluate(self, embeddings: np.ndarray, labels: List[Any]) -> Dict[str, float]:
        scores = cosine_similarity(
            np.array(embeddings).reshape(-1, embeddings[0].shape[-1]),
            np.array(embeddings).reshape(-1, embeddings[0].shape[-1])
        )
        y = np.array(labels)
        labels = y.reshape(-1, 1) == y.reshape(1, -1)
        scores = scores[np.triu_indices(scores.shape[0], k=1)]
        labels = labels[np.triu_indices(labels.shape[0], k=1)]

        eer = calculate_eer(labels, scores)
        auc = roc_auc_score(labels, scores)
        auc_threshold = roc_auc_score(labels, scores, max_fpr=0.01)

        return {
            "eer": eer,
            "auc": auc,
            "auc_threshold": auc_threshold
        }
