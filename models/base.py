from abc import ABC, abstractmethod
from typing import List
import numpy as np

class STEBModel(ABC):
    @abstractmethod
    def embed_single(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed a list of single texts."""
        pass

    @abstractmethod
    def embed_multiple(self, episodes: List[List[str]], batch_size: int) -> np.ndarray:
        """Embed a list of episodes (lists of texts)."""
        pass
