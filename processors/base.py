from abc import ABC, abstractmethod
from typing import Any, Dict, List

class Processor(ABC):
    @abstractmethod
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        pass
