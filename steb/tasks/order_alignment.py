from typing import Dict, Any, List, Hashable
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from .base import Task


# ---------- Helpers ----------

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize the last dimension of embeddings, safely."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def group_indices_by_label(labels: List[Hashable]) -> Dict[Hashable, List[int]]:
    """Return a mapping from label -> list of indices with that label."""
    label_to_indices: Dict[Hashable, List[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    return label_to_indices


def align_and_score(emb_src: np.ndarray, emb_tgt: np.ndarray) -> Dict[str, float]:
    """
    Align positions from emb_src to emb_tgt with Hungarian algorithm and compute
    alignment accuracy. emb_src has shape (n_src, dim), emb_tgt (n_tgt, dim).
    Assumes cosine similarity (embeddings already normalized).
    """
    n_src = emb_src.shape[0]
    n_tgt = emb_tgt.shape[0]

    # Need at least 2 positions in target to have meaningful ordering
    if n_tgt < 2:
        return {"accuracy": 0.0}

    # cosine since normalized
    sim_matrix = emb_src @ emb_tgt.T
    sim_matrix = np.maximum(sim_matrix, 0.0)
    cost_matrix = 1.0 - sim_matrix

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # sort rows so they correspond to positions 0..n_src-1
    order = np.argsort(row_indices)
    predicted_positions = col_indices[order]

    true_positions = np.arange(n_src)
    accuracy = float(np.mean(predicted_positions == true_positions))

    return {"accuracy": accuracy}


# ---------- Task ----------

class OrderAlignmentTask(Task):
    """
    A task for evaluating order alignment performance.
    Measures how well embeddings preserve the ordering of style intensity levels.
    Includes a distractor variant where the least-intense item from i is moved
    into j and removed from i.
    """
    def evaluate(self, embeddings: np.ndarray, labels: List[Hashable]) -> Dict[str, float]:
        """
        Args:
            embeddings: The embeddings to evaluate.
                        Expected format: shape (num_text_lists, num_positions, dim)
                        or a list of arrays of shape (num_positions, dim).
                        Positions represent ordered style levels (most -> least).
            labels: The corresponding labels. Only text_lists with matching labels are compared.

        Returns:
            A dictionary containing:
                - alignment_accuracy_mean
                - spearman_mean
                - distractor_accuracy_mean
                - distractor_spearman_mean
        """
        # Ensure ndarray
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=float)

        # Pre-normalize
        embeddings_norm = l2_normalize(embeddings)

        # Group by label
        label_to_indices = group_indices_by_label(labels)

        alignment_accuracies: List[float] = []
        distractor_last_accuracies: List[float] = []
        distractor_first_accuracies: List[float] = []

        # Iterate per label group
        for _, idxs in label_to_indices.items():
            if len(idxs) < 2:
                continue

            # All unordered pairs within label group
            for i, j in itertools.combinations(idxs, 2):
                emb_i = embeddings_norm[i]
                emb_j = embeddings_norm[j]

                # --- Baseline (no distractor), full lists ---
                base_scores = align_and_score(emb_i, emb_j)
                alignment_accuracies.append(base_scores["accuracy"])

                # --- Distractor variants ---
                if emb_i.shape[0] < 2:
                    continue

                # TODO: this is not symmetric, would we want to change sth about itertools.combinations?
                # Distractor variant 1: Replace last (least-intense) position
                emb_i_ref_last = emb_i[:-1]             # i without its last (least-intense) item
                emb_j_distr_last = emb_j.copy()
                emb_j_distr_last[-1] = emb_i[-1]        # j's last replaced by i's last

                distr_last_scores = align_and_score(emb_i_ref_last, emb_j_distr_last)
                distractor_last_accuracies.append(distr_last_scores["accuracy"])

                # Distractor variant 2: Replace first (most-intense) position
                emb_i_ref_first = emb_i[1:]             # i without its first (most-intense) item
                emb_j_distr_first = emb_j.copy()
                emb_j_distr_first[0] = emb_i[0]         # j's first replaced by i's first

                distr_first_scores = align_and_score(emb_i_ref_first, emb_j_distr_first)
                distractor_first_accuracies.append(distr_first_scores["accuracy"])

        # Calculate mean of both distractor variants
        all_distractor_accs = distractor_last_accuracies + distractor_first_accuracies

        return {
            "acc_mean": float(np.mean(alignment_accuracies)) if alignment_accuracies else 0.0,
            "distractor_last_acc_mean": float(np.mean(distractor_last_accuracies)) if distractor_last_accuracies else 0.0,
            "distractor_first_acc_mean": float(np.mean(distractor_first_accuracies)) if distractor_first_accuracies else 0.0,
            "distractor_acc_mean": float(np.mean(all_distractor_accs)) if all_distractor_accs else 0.0,
        }