from abc import ABC, abstractmethod
from typing import Any, List


class Processor(ABC):
    """
    An abstract base class for data processors.
    """

    @abstractmethod
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        """
        Processes the given embeddings and labels.

        Args:
            embeddings: The embeddings to process.
            labels: The corresponding labels.

        Returns:
            The processed data.
        """
        pass
