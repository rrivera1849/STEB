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
            # For single text episodes, use the text length as a simple feature
            text = episode[0] if isinstance(episode, list) else episode
            # Create a simple embedding based on text characteristics
            length_feature = len(text) / 100.0
            embeddings.append(np.array([length_feature, length_feature * 0.5]))
        return np.array(embeddings)

def test_order_alignment():
    """
    Test the order_alignment task with dummy data.
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

    # Collect all data from records
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

        # Embed both sets
        ordered_embeddings = model.embed_multiple([[text] for text in ordered_texts], batch_size=32)
        unordered_embeddings = model.embed_multiple([[text] for text in unordered_texts], batch_size=32)

        all_ordered_embeddings.append(ordered_embeddings)
        all_unordered_embeddings.append(unordered_embeddings)
        all_true_indices.append(true_indices)

    # Process with processor (should handle multiple records and aggregate)
    from processors.order_alignment import OrderAlignmentProcessor
    processor = OrderAlignmentProcessor()
    processed_data = processor.process(all_ordered_embeddings, all_unordered_embeddings, all_true_indices)

    # Evaluate with task (should return aggregated metrics)
    from tasks.order_alignment import OrderAlignmentTask
    task = OrderAlignmentTask()
    metrics = task.evaluate(*processed_data)

    print(f"\nOverall Metrics: {metrics}")

    # Verify expected metrics are present
    assert "spearman_correlation" in metrics, "Should have spearman_correlation"
    assert isinstance(metrics["spearman_correlation"], (float, np.floating)), "Correlation should be a float"

    # Check that correlation is in valid range [-1, 1]
    assert -1.0 <= metrics["spearman_correlation"] <= 1.0, f"Correlation {metrics['spearman_correlation']} out of range"

    print("✓ Order alignment test passed!")
    print(f"✓ Overall Spearman correlation: {metrics['spearman_correlation']:.3f}")
    return metrics

if __name__ == "__main__":
    test_order_alignment()
