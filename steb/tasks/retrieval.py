from typing import Dict, Any, List
import numpy as np
from .base import Task
from ..metrics import l2_normalize, calculate_retrieval_metrics

class RetrievalTask(Task):
    """
    A task for evaluating retrieval performance.
    """
    def evaluate(
        self, 
        embeddings: np.ndarray,
        labels: List[Any],
    ) -> Dict[str, float]:
        """
        Evaluates retrieval performance.

        Args:
            embeddings: Expected to be size of (num_embeddings, dim)
            labels: Expected to be size of (num_embeddings,)
        
        NOTE:
            1. Labels need to follow the format "N_query" and "N_target", where N is a 
               number convertible to an integer. Fortunately, users don't need to 
               manually ensure this format, as a built-in processor class handles all 
               the necessary pre-processing (see README).
            2. This task is designed to accommodate any quantity of true matches within the target set.

        Returns:
            Dictionary with MRR, Mean Rank, and Recall@K metrics.
        """

        embeddings = l2_normalize(embeddings)
        embeddings_query, embeddings_target = [], []
        query_labels, target_labels = [], []
        ii = 0
        for label in labels:
            if "_query" in label:
                embeddings_query.append(embeddings[ii])
                query_labels.append(int(label.split("_query")[0]))
            elif "_target" in label:
                embeddings_target.append(embeddings[ii])
                target_labels.append(int(label.split("_target")[0]))
            ii += 1
        embeddings_query = np.concatenate(embeddings_query, axis=0)
        embeddings_target = np.concatenate(embeddings_target, axis=0)
        query_labels = np.array(query_labels)
        target_labels = np.array(target_labels)

        # Filter to only include labels present in both sets
        common_labels = np.intersect1d(query_labels, target_labels)
        
        valid_query_indices = np.isin(query_labels, common_labels)
        embeddings_query = embeddings_query[valid_query_indices]
        query_labels = query_labels[valid_query_indices]

        valid_target_indices = np.isin(target_labels, common_labels)
        embeddings_target = embeddings_target[valid_target_indices]
        target_labels = target_labels[valid_target_indices]

        return calculate_retrieval_metrics(
            embeddings_query=embeddings_query,
            embeddings_target=embeddings_target,
            query_labels=query_labels,
            target_labels=target_labels
        )
