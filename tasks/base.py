from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class Task(ABC):
    @abstractmethod
    def evaluate(self, embeddings: np.ndarray, labels: List[Any]) -> Dict[str, float]:
        pass
