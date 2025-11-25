from typing import Dict, Any, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from .base import Task


class OrderAlignmentTask(Task):
    """
    A task for evaluating order alignment performance.
    Measures how well embeddings preserve the ordering of style intensity levels.
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Any]) -> Dict[str, float]:
        """
        Evaluates order preservation by comparing pairs of text lists.

        Args:
            embeddings: The embeddings to evaluate. Expected format: [[pos0_emb, pos1_emb, ...], ...]
                       where positions represent ordered style levels (most -> least)
                       Every text_list is evaluated w.r.t. every other text_list coming after it
            labels: The corresponding labels. Only text_lists with matching labels are compared.

        Returns:
            A dictionary containing order alignment metrics.
        """
        alignment_accuracies = []
        spearman_scores = []

        num_text_lists = len(embeddings)

        # Compare each text_list with every text_list that comes after it
        # Only compare text_lists with the same label
        for i in range(num_text_lists):
            for j in range(i + 1, num_text_lists):
                # Skip if labels don't match
                if labels[i] != labels[j]:
                    continue

                text_list_i = np.array(embeddings[i])
                text_list_j = np.array(embeddings[j])

                num_positions = len(text_list_i)
                if num_positions < 2:
                    continue

                # Compute similarity matrix between all positions of text_list_i and text_list_j
                # sim_matrix[k, l] = similarity between pos_k of text_list_i and pos_l of text_list_j
                sim_matrix = cosine_similarity(text_list_i, text_list_j)

                # Clamp negative similarities to 0
                sim_matrix = np.maximum(sim_matrix, 0)

                # Use linear sum assignment to find optimal one-to-one matching
                # Convert similarity to distance as cost matrix
                cost_matrix = 1 - sim_matrix
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # row_indices should be [0, 1, 2, ...] since it's just the positions in text_list_i
                # col_indices contains the matched positions in text_list_j
                predicted_positions = col_indices
                true_positions = np.arange(num_positions)

                # Calculate alignment accuracy: how many positions correctly aligned
                correct_alignments = np.sum(predicted_positions == true_positions)
                alignment_accuracy = correct_alignments / num_positions
                alignment_accuracies.append(alignment_accuracy)

                # Calculate Spearman correlation between true and predicted positions
                spearman, _ = spearmanr(true_positions, predicted_positions)
                spearman_scores.append(spearman)

        return {
            "alignment_accuracy_mean": float(np.mean(alignment_accuracies)) if alignment_accuracies else 0.0,
            "spearman_mean": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
        }