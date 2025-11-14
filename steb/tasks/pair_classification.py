from typing import Dict, Any, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from .base import Task

def calculate_eer(y_true, y_score):
    """
    Calculates the Equal Error Rate (EER).

    Args:
        y_true: The true labels.
        y_score: The predicted scores.

    Returns:
        The EER score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


class PairClassificationTask(Task):
    """
    A task for evaluating pair classification performance.
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Any]) -> Dict[str, float]:
        """
        Evaluates the performance of a pair classification model using EER and AUC.

        Args:
            embeddings: The embeddings to evaluate.
            labels: The corresponding labels.

        Returns:
            A dictionary of evaluation metrics, including EER, AUC, and AUC at various FPR thresholds.
        """
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

        return_d = {
            "eer": eer,
            "auc": auc,
        }
        for fpr in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
            auc_threshold = roc_auc_score(labels, scores, max_fpr=fpr)
            return_d["auc@{:.2f}".format(fpr)] = auc_threshold

        return return_d
