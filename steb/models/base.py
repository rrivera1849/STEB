from abc import ABC, abstractmethod
from typing import List

import numpy as np


class STEBModel(ABC):
    """
    Abstract base class for style text embedding models.
    """
    @abstractmethod
    def embed_multiple(self, episodes: List[List[str]], batch_size: int, show_progress: bool = False) -> np.ndarray:
        """
        Embeds a list of episodes, where each episode is a list of texts.

        Args:
            episodes: A list of episodes to embed.
            batch_size: The batch size to use for embedding.
            show_progress: Whether to show a progress bar.

        Returns:
            A numpy array of embeddings.
        """
        pass
