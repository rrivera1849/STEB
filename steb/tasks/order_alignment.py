from typing import Dict, Any, List, Hashable
import numpy as np
from scipy.optimize import linear_sum_assignment
from .base import Task


class OrderAlignmentTask(Task):
    """
    A task for evaluating order alignment performance.
    Measures how well embeddings preserve the ordering of style intensity levels.
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Hashable]) -> Dict[str, float]:
        """
        Evaluates order preservation by comparing pairs of text lists.

        Args:
            embeddings: The embeddings to evaluate.
                        Expected format: shape (num_text_lists, num_positions, dim)
                        or a list of arrays of shape (num_positions, dim).
                        Positions represent ordered style levels (most -> least).
            labels: The corresponding labels. Only text_lists with matching labels are compared.

        Returns:
            A dictionary containing order alignment metrics.
        """
        # Convert to ndarray if it's a list
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=float)

        # L2-normalize embeddings once so cosine similarity is just a dot product
        # shape: (num_lists, num_positions, dim)
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        embeddings_norm = embeddings / norms

        # Group indices by label to avoid O(N^2) label comparisons
        label_to_indices: Dict[Hashable, List[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)

        alignment_accuracies: List[float] = []
        spearman_scores: List[float] = []

        # Helper: fast Spearman for permutation
        def spearman_for_perm(predicted_positions: np.ndarray) -> float:
            """
            predicted_positions: permutation of [0, 1, ..., n-1]
            true positions are [0, 1, ..., n-1]
            Spearman rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
            where d_i = i - predicted_positions[i]
            """
            n = predicted_positions.size
            if n < 2:
                return 0.0
            diffs = np.arange(n) - predicted_positions
            sq_sum = float(np.sum(diffs * diffs))
            return 1.0 - (6.0 * sq_sum) / (n * (n * n - 1.0))

        # For each label, compare all pairs within that label group
        for label, idxs in label_to_indices.items():
            if len(idxs) < 2:
                continue

            # Pairwise comparisons within the group
            for a in range(len(idxs)):
                i = idxs[a]
                emb_i = embeddings_norm[i]
                num_positions_i = emb_i.shape[0]
                if num_positions_i < 2:
                    continue

                true_positions = np.arange(num_positions_i)

                for b in range(a + 1, len(idxs)):
                    j = idxs[b]
                    emb_j = embeddings_norm[j]

                    # Compute similarity matrix via dot product (cosine, since normalized)
                    # shape: (num_positions_i, num_positions_j)
                    sim_matrix = emb_i @ emb_j.T

                    # Clamp negative similarities to 0 (if desired)
                    sim_matrix = np.maximum(sim_matrix, 0.0)

                    # Hungarian algorithm on cost matrix
                    cost_matrix = 1.0 - sim_matrix
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    # We expect row_indices = [0..num_positions_i-1] if sizes match,
                    # but in case of unequal sizes, restrict to the first list's positions
                    # matched rows must be sorted to align with true_positions
                    order = np.argsort(row_indices)
                    predicted_positions = col_indices[order]

                    # Alignment accuracy
                    correct_alignments = np.sum(predicted_positions == true_positions)
                    alignment_accuracy = correct_alignments / float(num_positions_i)
                    alignment_accuracies.append(alignment_accuracy)

                    # Spearman correlation (fast formula)
                    spearman = spearman_for_perm(predicted_positions)
                    spearman_scores.append(spearman)

        return {
            "alignment_accuracy_mean": float(np.mean(alignment_accuracies)) if alignment_accuracies else 0.0,
            "spearman_mean": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
        }