from typing import Any, List
import numpy as np
from processors.base import Processor

class OrderAlignmentProcessor(Processor):
    """
    A processor for order alignment tasks.
    This processor handles ordered reference embeddings, unordered test embeddings,
    and true indices for alignment evaluation.
    """
    def process(self, ordered_embeddings: np.ndarray, unordered_embeddings: np.ndarray,
                true_indices: List[int]) -> tuple:
        """
        Processes the embeddings and indices for order alignment.

        Args:
            ordered_embeddings: Embeddings of the ordered reference set.
            unordered_embeddings: Embeddings of the unordered test set.
            true_indices: The true order indices of the unordered texts.

        Returns:
            A tuple of (ordered_embeddings, unordered_embeddings, true_indices).
        """
        return ordered_embeddings, unordered_embeddings, true_indices