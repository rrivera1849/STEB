from typing import Any, Dict, List


def load_dummy_clustering_dataset(_path: str) -> List[Dict[str, Any]]:
    """
    Provides a minimal synthetic dataset for clustering and pair classification tests.

    Generates 200 records across 2 classes (100 each), with distinct text
    patterns per class so that a reasonable embedding model can separate them.

    Args:
        path: Unused, kept for loader interface compatibility.

    Returns:
        A list of records with 'text' and 'label' fields.
    """
    records = []
    for i in range(100):
        records.append({
            "text": f"The weather today is sunny and warm, a perfect day number {i}.",
            "label": "weather",
        })
    for i in range(100):
        records.append({
            "text": f"Breaking news: the stock market index rose sharply in session {i}.",
            "label": "finance",
        })
    return records
