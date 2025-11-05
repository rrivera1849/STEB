from typing import Any, Dict, List
from processors.base import Processor

class PairClassificationProcessor(Processor):
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        return embeddings, labels
