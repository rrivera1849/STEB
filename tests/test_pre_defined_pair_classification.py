import numpy as np
from steb.tasks.pre_defined_pair_classification import PreDefinedPairClassificationTask

def test_pre_defined_pair_classification_perfect_metrics():
    """
    Test that the task returns perfect metrics (AUC=1.0, EER=0.0) when 
    positive pairs correspond to identical embeddings and negative pairs to orthogonal/opposite ones.
    """
    # 5 Positive pairs (High similarity)
    pos_pairs = []
    for _ in range(5):
        vec = np.random.rand(10)
        pos_pairs.append([vec, vec]) # Sim = 1.0 (float precision aside)
        
    # 5 Negative pairs (Low similarity)
    neg_pairs = []
    for _ in range(5):
        v1 = np.random.rand(10)
        # Make v2 orthogonal-ish or just different
        v2 = -v1 
        neg_pairs.append([v1, v2]) # Sim = -1.0
        
    # Construct the full input list
    # Input to task.evaluate is (embeddings, labels)
    # embeddings is a list of episodes.
    
    embeddings = []
    labels = []
    
    # Add positive pairs
    for i, (v1, v2) in enumerate(pos_pairs):
        # We need two separate episodes for each pair, sharing the same label
        # Each episode is a list of positions. Here we assume 1 position.
        # So episode = [embedding_vector]
        embeddings.append([v1])
        embeddings.append([v2])
        
        label = f"trial_{i}_true"
        labels.append(label)
        labels.append(label)
        
    # Add negative pairs
    for i, (v1, v2) in enumerate(neg_pairs):
        embeddings.append([v1])
        embeddings.append([v2])
        
        # Use a distinct ID for negatives relative to positives to avoid collision if IDs matter, 
        # though here we just need unique per pair.
        label = f"trial_{len(pos_pairs) + i}_false"
        labels.append(label)
        labels.append(label)
        
    task = PreDefinedPairClassificationTask()
    metrics = task.evaluate(embeddings, labels)
    
    assert metrics["auc"] == 1.0
    assert metrics["eer"] == 0.0