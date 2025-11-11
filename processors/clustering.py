from typing import Any, List
from processors.base import Processor

class ClusteringProcessor(Processor):
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        return embeddings, labels
