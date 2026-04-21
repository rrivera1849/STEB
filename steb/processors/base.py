from typing import Any, List, Tuple


class Processor:
    """
    Base data processor. Returns embeddings and labels as-is.
    Subclass and override process() for custom behavior.
    """

    def process(
        self,
        embeddings: Any,
        labels: List[Any],
    ) -> Tuple[Any, List[Any]]:
        """
        Processes the given embeddings and labels.

        Args:
            embeddings: The embeddings to process.
            labels: The corresponding labels.

        Returns:
            A tuple of the embeddings and labels.
        """
        return embeddings, labels
