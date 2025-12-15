from typing import Any, List
from .base import Processor

class AllToAllPairClassificationProcessor(Processor):
    """
    A processor for all-to-all pair classification tasks.
    This processor simply returns the embeddings and labels as is.
    """
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        """
        Processes the given embeddings and labels for pair classification.

        Args:
            embeddings: The embeddings to process.
            labels: The corresponding labels.

        Returns:
            A tuple of the embeddings and labels.
        """
        return embeddings, labels
