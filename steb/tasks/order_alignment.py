import itertools
from typing import Any, Dict, Hashable, List, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..metrics import l2_normalize
from .base import Task


# ---------- Helpers ----------

def group_indices_by_label(labels: List[Hashable]) -> Dict[Hashable, List[int]]:
    """Return a mapping from label -> list of indices with that label."""
    label_to_indices: Dict[Hashable, List[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    return label_to_indices


def align_and_score(
    emb_src: np.ndarray,
    emb_tgt: np.ndarray,
    offset: int = 0,
) -> Dict[str, float]:
    """
    Align positions from emb_src to emb_tgt with Hungarian algorithm and compute
    alignment accuracy. emb_src has shape (n_src, dim), emb_tgt (n_tgt, dim).
    Assumes cosine similarity (embeddings already normalized).

    Args:
        emb_src: Source embeddings to align.
        emb_tgt: Target embeddings to align to.
        offset: Position offset for expected alignment (e.g., if src starts at position 1, offset=1).
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

    # Expected positions account for offset (e.g., [1, 2, 3] if offset=1)
    true_positions = np.arange(n_src) + offset
    accuracy = float(np.mean(predicted_positions == true_positions))

    return {"accuracy": accuracy}


# ---------- Task ----------

class OrderAlignmentTask(Task):
    """
    A task for evaluating order alignment performance.
    Measures how well embeddings preserve the ordering of style intensity levels when using the Hungarian algorithm for alignment.
    Includes distractor variants where either the least- or most-intense item from the first list is moved into the second list and removed from the first list.
    """
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: List[Any],
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Args:
            embeddings: The embeddings to evaluate.
                        Expected format: shape (num_text_lists, num_positions, dim)
                        or a list of arrays of shape (num_positions, dim).
                        Positions represent ordered style levels (most -> least).
            labels: The corresponding labels. Only text_lists with matching labels are compared.

        Returns:
            A dictionary containing:
                - acc_mean: Mean alignment accuracy across all pairs of text lists with the same label.
                - distractor_acc_mean: Mean alignment accuracy under the distractor setting (combining the distractor variants).
                - _per_label: Mapping from label (as string) to its own
                    {acc_mean, distractor_acc_mean} computed only over within-label pair comparisons.
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

        per_label_alignment_accs: Dict[Hashable, List[float]] = {}
        per_label_distractor_accs: Dict[Hashable, List[float]] = {}

        # Iterate per label group
        for label, idxs in label_to_indices.items():
            if len(idxs) < 2:
                continue

            # All unordered pairs within label group
            for i, j in itertools.combinations(idxs, 2):  # note: itertools.combinations is deterministic
                emb_i = embeddings_norm[i]
                emb_j = embeddings_norm[j]

                # --- Baseline (no distractor), full lists ---
                base_scores = align_and_score(emb_i, emb_j)
                alignment_accuracies.append(base_scores["accuracy"])
                per_label_alignment_accs.setdefault(label, []).append(base_scores["accuracy"])

                # --- Distractor variants ---
                # NOTE: distractor calculations are not symmetric
                #   This is the case because the distractor manipulations are not symmetric,
                #       and we are not exhaustive in pair selection (itertools.combinations).
                #       This is done to reduce and avoid exponential compute.
                #       However, we expect the averaged scores not to change dramatically.
                #   The results remain comparable for the same settings as itertools.combinations is deterministic.
                #   BUT if one would go and change the ordering of elements in the datasets,
                #       this would potentially make results not comparable.
                if emb_i.shape[0] < 2:
                    continue

                # Distractor variant 1: Replace last (least-intense) position
                emb_i_ref_last = emb_i[:-1]             # i without its last (least-intense) item
                emb_j_distr_last = emb_j.copy()
                emb_j_distr_last[-1] = emb_i[-1]        # j's last replaced by i's last

                distr_last_scores = align_and_score(emb_i_ref_last, emb_j_distr_last)
                distractor_last_accuracies.append(distr_last_scores["accuracy"])
                per_label_distractor_accs.setdefault(label, []).append(distr_last_scores["accuracy"])

                # Distractor variant 2: Replace first (most-intense) position
                emb_i_ref_first = emb_i[1:]             # i without its first (most-intense) item
                emb_j_distr_first = emb_j.copy()
                emb_j_distr_first[0] = emb_i[0]         # j's first replaced by i's first

                distr_first_scores = align_and_score(emb_i_ref_first, emb_j_distr_first, offset=1)
                distractor_first_accuracies.append(distr_first_scores["accuracy"])
                per_label_distractor_accs.setdefault(label, []).append(distr_first_scores["accuracy"])

        # Calculate mean of both distractor variants
        all_distractor_accs = distractor_last_accuracies + distractor_first_accuracies

        per_label = {
            str(label): {
                "acc_mean": float(np.mean(per_label_alignment_accs[label]))
                            if per_label_alignment_accs.get(label) else 0.0,
                "distractor_acc_mean": float(np.mean(per_label_distractor_accs[label]))
                                       if per_label_distractor_accs.get(label) else 0.0,
            }
            for label in per_label_alignment_accs
        }

        return {
            "acc_mean": float(np.mean(alignment_accuracies)) if alignment_accuracies else 0.0,
            "distractor_acc_mean": float(np.mean(all_distractor_accs)) if all_distractor_accs else 0.0,
            "_per_label": per_label,
        }