from typing import Any, List
from .base import Processor

class ClusteringProcessor(Processor):
    """
    A processor for clustering tasks.
    This processor simply returns the embeddings and labels as is.
    """
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        """
        Processes the given embeddings and labels for clustering.

        Args:
            embeddings: The embeddings to process.
            labels: The corresponding labels.

        Returns:
            A tuple of the embeddings and labels.
        """
        return embeddings, labels
