import sys
import numpy as np
from scipy.stats import spearmanr

# Mock model for testing
class MockModel:
    def embed_multiple(self, episodes, batch_size):
        """
        Mock embedding that creates simple vectors based on text length.
        Shorter texts get smaller values, longer texts get larger values.
        This simulates formality increasing with text length.
        """
        embeddings = []
        for episode in episodes:
            text = episode[0] if isinstance(episode, list) else episode
            length_feature = len(text) / 100.0
            embeddings.append(np.array([length_feature, 1 - length_feature]))
        return np.array(embeddings)

def test_order_alignment():
    """
    Test the order_alignment task with dummy data including distractors.
    Should return aggregated metrics across all test cases.
    """
    print("Testing order_alignment task...")

    # Load the dummy dataset
    from steb_datasets.dummy_order_alignment.loader import load_dummy_order_alignment_dataset
    records = load_dummy_order_alignment_dataset("./steb_datasets/dummy_order_alignment")

    print(f"Loaded {len(records)} test cases")
    assert len(records) > 0, "Should have at least one record"

    # Create mock model
    model = MockModel()

    all_ordered_embeddings = []
    all_unordered_embeddings = []
    all_true_indices = []

    for i, record in enumerate(records):
        print(f"\nProcessing test case {i+1}...")
        assert "ordered_texts" in record
        assert "unordered_texts" in record
        assert "true_indices" in record

        ordered_texts = record["ordered_texts"]
        unordered_texts = record["unordered_texts"]
        true_indices = record["true_indices"]

        ordered_embeddings = model.embed_multiple(ordered_texts, batch_size=32)
        unordered_embeddings = model.embed_multiple(unordered_texts, batch_size=32)

        all_ordered_embeddings.append(ordered_embeddings)
        all_unordered_embeddings.append(unordered_embeddings)
        all_true_indices.append(true_indices)

    from processors.order_alignment import OrderAlignmentProcessor
    processor = OrderAlignmentProcessor()
    processed_data = processor.process(all_ordered_embeddings, all_unordered_embeddings, all_true_indices)

    from tasks.order_alignment import OrderAlignmentTask
    task = OrderAlignmentTask()
    metrics = task.evaluate(*processed_data)

    print(f"\nOverall Metrics: {metrics}")

    assert "spearman_correlation" in metrics, "Should have spearman_correlation"
    assert "f1_score" in metrics, "Should have f1_score"
    assert isinstance(metrics["spearman_correlation"], (float, np.floating)), "Correlation should be a float"
    assert isinstance(metrics["f1_score"], (float, np.floating)), "F1 score should be a float"

    assert -1.0 <= metrics["spearman_correlation"] <= 1.0, f"Correlation {metrics['spearman_correlation']} out of range"
    assert 0.0 <= metrics["f1_score"] <= 1.0, f"F1 score {metrics['f1_score']} out of range"

    print("✓ Order alignment test passed!")
    print(f"✓ Overall Spearman correlation: {metrics['spearman_correlation']:.3f}")
    print(f"✓ Overall F1 score: {metrics['f1_score']:.3f}")
    return metrics

if __name__ == "__main__":
    test_order_alignment()
