from typing import Dict, List
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tasks.base import Task

class OrderAlignmentTask(Task):
    """
    A task for evaluating order alignment performance with distractors.
    """
    def evaluate(self, ordered_embeddings: List[np.ndarray], unordered_embeddings: List[np.ndarray],
                 true_indices: List[List[int]]) -> Dict[str, float]:
        """
        Evaluates the performance of an order alignment model using Spearman correlation and F1 score.

        Args:
            ordered_embeddings: A list of arrays, where each array is the embeddings of the ordered reference set.
            unordered_embeddings: A list of arrays, where each array is the embeddings of the unordered test set with distractors.
            true_indices: A list of lists, where each inner list contains the true order indices of the unordered texts, with -1 for distractors.

        Returns:
            A dictionary of evaluation metrics, including the mean Spearman correlation and F1 score.
        """
        correlations = []
        f1_scores = []

        for i in range(len(ordered_embeddings)):
            similarity_matrix = cosine_similarity(unordered_embeddings[i], ordered_embeddings[i])

            # Select the top N embeddings from the unordered set, where N is the number of ordered embeddings
            predicted_indices_in_unordered = np.argmax(similarity_matrix, axis=0)

            # Get the true indices for the selected items
            selected_true_indices = [true_indices[i][j] for j in predicted_indices_in_unordered]

            # Identify which of the selected items are not distractors
            y_true = [1 if idx != -1 else 0 for idx in true_indices[i]]
            y_pred = [0] * len(true_indices[i])
            for j in predicted_indices_in_unordered:
                y_pred[j] = 1

            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)

            # Filter out distractors for Spearman correlation
            non_distractor_true = [idx for idx in selected_true_indices if idx != -1]
            non_distractor_pred = [j for j, idx in enumerate(selected_true_indices) if idx != -1]

            if len(non_distractor_true) > 1:
                correlation, _ = spearmanr(non_distractor_true, non_distractor_pred)
                correlations.append(correlation)

        return {
            "spearman_correlation": np.mean(correlations) if correlations else 0.0,
            "f1_score": np.mean(f1_scores) if f1_scores else 0.0
        }
