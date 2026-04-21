from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class Task(ABC):
    """
    An abstract base class for evaluation tasks.
    """

    @abstractmethod
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: List[Any],
    ) -> Dict[str, float]:
        """
        Evaluates the given embeddings and labels.

        Args:
            embeddings: The embeddings to evaluate.
            labels: The corresponding labels.

        Returns:
            A dictionary of evaluation metrics.
        """
        pass
