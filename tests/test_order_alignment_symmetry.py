"""Tests for order alignment symmetry and concrete distractor-variant behavior."""
import itertools

import numpy as np
import pytest

from steb.tasks.order_alignment import OrderAlignmentTask, align_and_score
from steb.metrics import l2_normalize


# ---------------------------------------------------------------------------
# 1. Baseline symmetry: align_and_score(A, B) == align_and_score(B, A)
# ---------------------------------------------------------------------------


class TestBaselineSymmetry:
    """Verify that the baseline (no-distractor) alignment is symmetric.

    Mathematically, transposing the cost matrix produces the inverse
    permutation, which has the same set of fixed points.  Therefore
    accuracy must be identical regardless of argument order.
    """

    @staticmethod
    def _random_embeddings(n_positions, dim, rng):
        emb = rng.standard_normal((n_positions, dim))
        return l2_normalize(emb)

    @pytest.mark.parametrize("n_positions", [2, 3, 5, 8])
    def test_symmetry_random(self, n_positions):
        """Random embeddings: score(A, B) == score(B, A)."""
        rng = np.random.default_rng(42)
        emb_a = self._random_embeddings(n_positions, 16, rng)
        emb_b = self._random_embeddings(n_positions, 16, rng)

        score_ab = align_and_score(emb_a, emb_b)["accuracy"]
        score_ba = align_and_score(emb_b, emb_a)["accuracy"]
        assert score_ab == pytest.approx(score_ba), (
            f"Baseline alignment should be symmetric: "
            f"score(A,B)={score_ab} != score(B,A)={score_ba}"
        )


# ---------------------------------------------------------------------------
# 2. Distractor variants: verify behavior on concrete 3-position examples
# ---------------------------------------------------------------------------


class TestDistractorVariants:
    """Verify distractor variants using 3-position examples with explicit shapes.

    Each text list has 3 positions ordered by style intensity:
        position 0 = most intense   (e.g. very formal)
        position 1 = medium
        position 2 = least intense  (e.g. very informal)

    Each position is an embedding vector of dimension `dim`.
    So one text list is a matrix of shape (3, dim).

    The task compares pairs of text lists (I, J) that share the same label:

    Baseline:
        align_and_score(I, J)  where I is (3, dim) and J is (3, dim).
        This builds a 3×3 cost matrix and finds the best one-to-one matching.
        Perfect result: I[0]→J[0], I[1]→J[1], I[2]→J[2].

    Distractor variant 1 ("last"):
        - Remove I[2] (least intense) from I    → I_ref  has shape (2, dim)
        - Replace J[2] with I[2] in J           → J_dist has shape (3, dim)
        - align_and_score(I_ref, J_dist) builds a 2×3 rectangular cost matrix.
          Only 2 of the 3 columns get matched. The distractor I[2] at col 2
          should be left unmatched if embeddings are good.

    Distractor variant 2 ("first"):
        - Remove I[0] (most intense) from I     → I_ref  has shape (2, dim)
        - Replace J[0] with I[0] in J           → J_dist has shape (3, dim)
        - align_and_score(I_ref, J_dist, offset=1) builds a 2×3 cost matrix.
          The distractor I[0] at col 0 should be left unmatched.
          offset=1 because the expected correct columns are now 1 and 2.
    """

    def test_distractor_last_3_positions(self):
        """Walk through the "last" distractor variant step by step.

        We use two DIFFERENT text lists I and J so the distractor injection
        actually changes J (unlike identity embeddings where I[2] == J[2]).

        I (3 positions, dim=4):
            pos 0 (most):  [1, 0, 0, 0]   — points along axis 0
            pos 1 (mid):   [0, 1, 0, 0]   — points along axis 1
            pos 2 (least): [0, 0, 1, 0]   — points along axis 2

        J (3 positions, dim=4):
            pos 0 (most):  [0.9, 0.1, 0, 0]  — close to I[0]
            pos 1 (mid):   [0.1, 0.9, 0, 0]  — close to I[1]
            pos 2 (least): [0, 0, 0.9, 0.1]  — close to I[2]

        Baseline: align_and_score(I, J)
            Shape: I is (3, 4), J is (3, 4) → 3×3 cost matrix.
            Best matching: I[0]→J[0], I[1]→J[1], I[2]→J[2] → accuracy = 1.0

        Distractor last:
            I_ref  = I[:-1] = [[1,0,0,0], [0,1,0,0]]             shape (2, 4)
            J_dist = J.copy(); J_dist[2] = I[2] = [0,0,1,0]      shape (3, 4)
              so J_dist = [[0.9,0.1,0,0], [0.1,0.9,0,0], [0,0,1,0]]

            align_and_score(I_ref, J_dist):
                sim_matrix (2×3):
                    I_ref[0]=[1,0,0,0] vs J_dist cols → [0.9, 0.1, 0.0]
                    I_ref[1]=[0,1,0,0] vs J_dist cols → [0.1, 0.9, 0.0]
                cost_matrix = 1 - sim_matrix (2×3):
                    [[0.1, 0.9, 1.0],
                     [0.9, 0.1, 1.0]]
                Hungarian picks: row 0→col 0 (cost 0.1), row 1→col 1 (cost 0.1)
                    → col 2 (the distractor) is unmatched.
                predicted = [0, 1], expected = [0, 1] → accuracy = 1.0
        """
        emb_i = l2_normalize(np.array([
            [1.0, 0.0, 0.0, 0.0],  # pos 0: most intense
            [0.0, 1.0, 0.0, 0.0],  # pos 1: medium
            [0.0, 0.0, 1.0, 0.0],  # pos 2: least intense
        ]))
        emb_j = l2_normalize(np.array([
            [0.9, 0.1, 0.0, 0.0],  # pos 0: close to I[0]
            [0.1, 0.9, 0.0, 0.0],  # pos 1: close to I[1]
            [0.0, 0.0, 0.9, 0.1],  # pos 2: close to I[2]
        ]))

        # Baseline: (3, 4) vs (3, 4) → 3×3 cost matrix
        baseline = align_and_score(emb_i, emb_j)
        assert baseline["accuracy"] == pytest.approx(1.0)

        # Distractor last:
        #   I_ref = I without pos 2 → shape (2, 4)
        emb_i_ref = emb_i[:-1]
        assert emb_i_ref.shape == (2, 4)

        #   J_dist = J with pos 2 replaced by I[2] → shape (3, 4)
        #   This changes J[2] from [0,0,0.9,0.1] to [0,0,1,0] — a real change.
        emb_j_dist = emb_j.copy()
        emb_j_dist[-1] = emb_i[-1]
        assert emb_j_dist.shape == (3, 4)

        #   align_and_score: (2, 4) vs (3, 4) → 2×3 rectangular cost matrix
        #   Only 2 of 3 columns get assigned. Col 2 (distractor) should be skipped.
        score = align_and_score(emb_i_ref, emb_j_dist, offset=0)
        assert score["accuracy"] == pytest.approx(1.0)
