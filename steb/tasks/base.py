from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

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
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluates the given embeddings and labels.

        Args:
            embeddings: The embeddings to evaluate.
            labels: The corresponding labels.

        Returns:
            A dictionary of evaluation metrics. Each value is either a
            scalar metric (float) or a nested dict mapping a sub-key
            (e.g. a label name) to its own scalar metrics.
        """
        pass
