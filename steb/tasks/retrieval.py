from collections import defaultdict
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import Task

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize the last dimension of embeddings, safely."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms

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

        sim_matrix = cosine_similarity(embeddings_query, embeddings_target)
        
        mrr_sum = 0.0
        rank_sum = 0.0
        recall_at_K = defaultdict(float)
        Ks = [1, 8, 16, 32, 64, 128] # RRS - Perhaps we should make this customizable?
        n_queries = query_labels.shape[0]
        
        for i in range(n_queries):
            query_label = query_labels[i]
            scores = sim_matrix[i]
            sorted_indices = np.argsort(-scores)
            ranks = np.where(query_label == target_labels[sorted_indices])[0]
            ranks += 1
            
            if ranks.size == 0:
                raise ValueError("No true matches found for query {}".format(i))
            
            mrr_sum += np.mean(1.0 / ranks)
            rank_sum += np.mean(ranks)
            for K in Ks:
                valid = np.where(ranks <= K)[0]
                if valid.size == 0:
                    recall_at_K[K] = 0.
                else:
                    recall_at_K[K] += valid.size / ranks.size

        metrics = {
            "mrr": mrr_sum / n_queries,
            "mean_rank": rank_sum / n_queries,
        }
        for K in Ks:
            metrics[f"recall@{K}"] = recall_at_K[K] / n_queries
        
        return metrics
