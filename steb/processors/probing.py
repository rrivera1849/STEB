import json
from typing import Any, List, Tuple

import numpy as np

from .base import Processor


class ProbingProcessor(Processor):
    def process(
        self,
        embeddings: List[List[np.ndarray]],
        labels: List[Any],
    ) -> Tuple[np.ndarray, List[Any]]:
        # embeddings contains shape: (num_texts, sequence_length_per_episode=1, 1, embedding_dim)
        # flatten to (num_texts, embedding_dim)
        X = np.array([e[0] for e in embeddings])
        # labels contains the JSON-serialized metadata 
        parsed_labels = [json.loads(l) for l in labels]
        return X, parsed_labels
