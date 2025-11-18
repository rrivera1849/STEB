from abc import ABC, abstractmethod
from typing import List
import numpy as np

class STEBModel(ABC):
    """
    An abstract base class for style text embedding models.
    """

    @abstractmethod
    def embed_single(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Embeds a list of single texts.

        Args:
            texts: A list of strings to embed.
            batch_size: The batch size to use for embedding.

        Returns:
            A numpy array of embeddings.
        """
        pass

    @abstractmethod
    def embed_multiple(self, episodes: List[List[str]], batch_size: int) -> np.ndarray:
        """
        Embeds a list of episodes, where each episode is a list of texts.

        Args:
            episodes: A list of episodes to embed.
            batch_size: The batch size to use for embedding.

        Returns:
            A numpy array of embeddings.
        """
        pass
