from typing import Any, List
from processors.base import Processor

class NoopProcessor(Processor):
    def process(self, embeddings: Any, labels: List[Any]) -> Any:
        return embeddings, labels
