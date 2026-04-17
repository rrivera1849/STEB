from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances, roc_auc_score, roc_curve, v_measure_score


def l2_normalize(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalizes the last dimension of embeddings.

    Args:
        embeddings: Array of embeddings to normalize.

    Returns:
        The L2-normalized embeddings (zero vectors are left unchanged).
    """
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms

def calculate_eer(
    y_true: np.ndarray, 
    y_score: np.ndarray,
) -> float:
    """
    Calculates the Equal Error Rate (EER).

    Args:
        y_true: The true labels.
        y_score: The predicted scores.

    Returns:
        The EER score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return float(eer)



DEFAULT_FPR_THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]


def calculate_pair_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Calculates EER, AUC, and AUC at specific FPR thresholds.

    Args:
        y_true: True labels.
        y_score: Predicted scores.
        fpr_thresholds: List of FPR thresholds for partial AUC. Defaults to common values.

    Returns:
        Dictionary of metrics.
    """
    if fpr_thresholds is None:
        fpr_thresholds = DEFAULT_FPR_THRESHOLDS

    eer = calculate_eer(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    metrics = {
        "eer": eer,
        "auc": auc,
    }

    for fpr in fpr_thresholds:
        metrics[f"auc@{fpr:.2f}"] = roc_auc_score(y_true, y_score, max_fpr=fpr)

    return metrics



def calculate_clustering_metrics(
    embeddings: np.ndarray,
    labels: List[Any],
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    distance_mode: str = "euclidean",
) -> Dict[str, float]:
    """
    Trains a K-Means model and evaluates its performance using V-measure.
    """
    if distance_mode not in {"euclidean", "l1_diff"}:
        raise ValueError(
            f"Unsupported distance_mode for clustering: {distance_mode}. "
            "Expected 'euclidean' or 'l1_diff'."
        )

    if embeddings.ndim == 1:
        embeddings = np.array(embeddings).reshape(-1, 1)

    if n_clusters is None:
        n_clusters = len(set(labels))

    # K-Means is Euclidean by design. For "l1_diff", apply an absolute-value
    # transform to better align with L1-style embedding differences while
    # keeping the same clustering backend.
    if distance_mode == "l1_diff":
        embeddings = np.abs(embeddings)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=32,
        n_init="auto",
    )
    kmeans.fit(embeddings)
    v_measure = v_measure_score(labels, kmeans.labels_)
    return {"v_measure": v_measure}



def calculate_retrieval_metrics(
    embeddings_query: np.ndarray,
    embeddings_target: np.ndarray,
    query_labels: np.ndarray,
    target_labels: np.ndarray,
    ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Calculates retrieval metrics: MRR, Mean Rank, Recall@K.
    """
    if ks is None:
        ks = [1, 8, 16, 32, 64, 128]

    dist_matrix = pairwise_distances(embeddings_query, embeddings_target, metric="cosine", n_jobs=-1)

    mrr_sum = 0.0
    rank_sum = 0.0
    recall_at_k = defaultdict(float)
    n_queries = query_labels.shape[0]

    for i in range(n_queries):
        query_label = query_labels[i]
        sorted_indices = np.argsort(dist_matrix[i])

        ranks = np.where(query_label == target_labels[sorted_indices])[0]
        ranks += 1  # 1-indexed

        if ranks.size == 0:
            raise ValueError(f"No true matches found for query {i}")

        mrr_sum += np.mean(1.0 / ranks)
        rank_sum += np.mean(ranks)

        for k in ks:
            valid = np.where(ranks <= k)[0]
            if valid.size > 0:
                recall_at_k[k] += valid.size / ranks.size

    metrics = {
        "mrr": mrr_sum / n_queries,
        "mean_rank": rank_sum / n_queries,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k[k] / n_queries

    return metrics
