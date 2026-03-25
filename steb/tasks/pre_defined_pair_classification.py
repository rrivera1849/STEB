from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..metrics import calculate_pair_classification_metrics
from .base import Task


class PreDefinedPairClassificationTask(Task):
    """
    A task for evaluating pair classification performance on pre-defined pairs.
    """
    def evaluate(self, embeddings: List[Any], labels: List[Any]) -> Dict[str, float]:
        """
        Evaluates the performance of a pair classification model using EER and AUC.
        
        Args:
            embeddings: List of episodes. 
            labels: The corresponding labels. Expected format "trial_N_true" or "trial_N_false".

        Returns:
            A dictionary of evaluation metrics, including EER, AUC, and AUC at various FPR thresholds.
        """
        grouped = {}
        for episode, label in zip(embeddings, labels):
            emb = episode[0]
            grouped.setdefault(label, []).append(emb)

        scores = []
        clean_labels = []

        for label, embs in grouped.items():
            if len(embs) != 2:
                raise ValueError(f"Expected 2 embeddings for label {label}, got {len(embs)}")

            e1 = embs[0].reshape(1, -1)
            e2 = embs[1].reshape(1, -1)

            sim = cosine_similarity(e1, e2)[0][0]
            scores.append(sim)

            if label.endswith("_true"):
                binary_label = 1
            elif label.endswith("_false"):
                binary_label = 0
            else:
                raise ValueError(f"Invalid label: {label}")

            clean_labels.append(binary_label)

        scores = np.array(scores)
        y = np.array(clean_labels)

        return calculate_pair_classification_metrics(y, scores)
