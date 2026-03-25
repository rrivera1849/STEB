from typing import Any, List

from .base import Processor


class PreDefinedPairClassificationProcessor(Processor):
    """
    A processor for pre-defined pair classification tasks.
    This processor takes embeddings and labels, assumes they are loaded as pairs (episode_size=2),
    and formats them for the task.
    """
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        # embeddings is a list of episodes.
        # If episode_size=2 was used during loading, each episode has 2 positions (texts).
        # We pass them as is, the task will handle the pairing logic.
        return embeddings, labels
