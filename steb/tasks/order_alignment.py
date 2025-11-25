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
            A dictionary containing order alignment metrics, including a
            1-distractor variant where the least-intense item from i is
            moved into j and removed from i.
        """
        # Convert to ndarray if it's a list
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=float)

        # L2-normalize embeddings once so cosine similarity is just a dot product
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        embeddings_norm = embeddings / norms

        # Group indices by label to avoid O(N^2) label comparisons
        label_to_indices: Dict[Hashable, List[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)

        alignment_accuracies: List[float] = []
        spearman_scores: List[float] = []

        distractor_alignment_accuracies: List[float] = []
        distractor_spearman_scores: List[float] = []

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

            for a in range(len(idxs)):
                i = idxs[a]
                emb_i = embeddings_norm[i]
                num_positions_i = emb_i.shape[0]
                if num_positions_i < 2:
                    continue

                true_positions_full = np.arange(num_positions_i)

                for b in range(a + 1, len(idxs)):
                    j = idxs[b]
                    emb_j = embeddings_norm[j]

                    # ---------- Original (no distractor) ----------
                    sim_matrix = emb_i @ emb_j.T  # cosine since normalized
                    sim_matrix = np.maximum(sim_matrix, 0.0)
                    cost_matrix = 1.0 - sim_matrix
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    # Sort rows to align with 0..num_positions_i-1
                    order = np.argsort(row_indices)
                    predicted_positions = col_indices[order]

                    correct_alignments = np.sum(predicted_positions == true_positions_full)
                    alignment_accuracy = correct_alignments / float(num_positions_i)
                    alignment_accuracies.append(alignment_accuracy)

                    spearman = spearman_for_perm(predicted_positions)
                    spearman_scores.append(spearman)

                    # ---------- 1-distractor variant ----------
                    # Move least-intense item from i into j and remove it from i.
                    # i_ref: positions 0..num_positions_i-2
                    emb_i_ref = emb_i[:-1]
                    num_positions_ref = emb_i_ref.shape[0]
                    if num_positions_ref == 0:
                        continue  # nothing to align

                    true_positions_ref = np.arange(num_positions_ref)

                    emb_j_distr = emb_j.copy()
                    # Replace j's least-intense position with i's least-intense embedding
                    emb_j_distr[-1] = emb_i[-1]

                    sim_matrix_distr = emb_i_ref @ emb_j_distr.T
                    sim_matrix_distr = np.maximum(sim_matrix_distr, 0.0)
                    cost_matrix_distr = 1.0 - sim_matrix_distr
                    row_indices_d, col_indices_d = linear_sum_assignment(cost_matrix_distr)

                    order_d = np.argsort(row_indices_d)
                    predicted_positions_d = col_indices_d[order_d]

                    correct_alignments_d = np.sum(predicted_positions_d == true_positions_ref)
                    alignment_accuracy_d = correct_alignments_d / float(num_positions_ref)
                    distractor_alignment_accuracies.append(alignment_accuracy_d)

                    spearman_d = spearman_for_perm(predicted_positions_d)
                    distractor_spearman_scores.append(spearman_d)

        return {
            "alignment_accuracy_mean": float(np.mean(alignment_accuracies)) if alignment_accuracies else 0.0,
            "spearman_mean": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
            "distractor_accuracy_mean": float(np.mean(distractor_alignment_accuracies)) if distractor_alignment_accuracies else 0.0,
            "distractor_spearman_mean": float(np.mean(distractor_spearman_scores)) if distractor_spearman_scores else 0.0,
        }