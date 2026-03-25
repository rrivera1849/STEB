from typing import Any, List, Tuple

import numpy as np

from .base import Processor


class RetrievalProcessor(Processor):
    def process(self, embeddings: List[List[np.ndarray]], labels: List[Any]) -> Tuple[np.ndarray, np.ndarray, List[Any], List[Any]]:
        return embeddings, labels
