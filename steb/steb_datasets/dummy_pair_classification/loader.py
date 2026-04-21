from typing import Any, Dict, List


def load_dummy_pair_classification_dataset(_path: str) -> List[Dict[str, Any]]:
    """
    Provides a minimal synthetic dataset for pre-defined pair classification tests.

    Generates 10 positive pairs (same-topic texts) and 10 negative pairs
    (different-topic texts), following the trial_N_true/trial_N_false label
    convention used by PAN and Fisher datasets.

    Args:
        path: Unused, kept for loader interface compatibility.

    Returns:
        A list of records with 'text' and 'label' fields.
    """
    records = []

    # Positive pairs: both texts about the same topic
    for i in range(10):
        records.append({
            "text": f"The sun is shining brightly today, it is a lovely warm day number {i}.",
            "label": f"trial_{i}_true",
        })
        records.append({
            "text": f"What a beautiful sunny day it is, warm and pleasant, day number {i}.",
            "label": f"trial_{i}_true",
        })

    # Negative pairs: texts about different topics
    for i in range(10):
        records.append({
            "text": f"The sun is shining brightly today, warm day number {i}.",
            "label": f"trial_{10 + i}_false",
        })
        records.append({
            "text": f"Quantum mechanics describes particle behavior at atomic scales, paper {i}.",
            "label": f"trial_{10 + i}_false",
        })

    return records
