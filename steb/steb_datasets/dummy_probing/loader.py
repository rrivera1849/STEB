import hashlib
import json
from typing import Any, Dict, List, Optional


def load_dummy_probing_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Provides a minimal synthetic dataset for probing task tests.

    Generates records with a single probing task (binary classification).
    Each record has text, a label list, and a split list following the
    probing dataset format.

    Args:
        path: Unused, kept for loader interface compatibility.

    Returns:
        A list of records with 'text', 'label', and 'split' fields.
    """
    records = []

    # Training set: 40 samples per class
    for i in range(40):
        records.append({
            "text": f"This is a short simple sentence number {i}.",
            "label": [0],
            "split": ["train"],
        })
        records.append({
            "text": f"The extraordinarily complex and multifaceted nature of this particular sentence number {i} is remarkable.",
            "label": [1],
            "split": ["train"],
        })

    # Validation set: 10 samples per class
    for i in range(10):
        records.append({
            "text": f"A brief sentence for validation {i}.",
            "label": [0],
            "split": ["val"],
        })
        records.append({
            "text": f"An exceedingly elaborate and thoroughly detailed validation sentence number {i} indeed.",
            "label": [1],
            "split": ["val"],
        })

    # Test set: 10 samples per class
    for i in range(10):
        records.append({
            "text": f"Short test text {i}.",
            "label": [0],
            "split": ["test"],
        })
        records.append({
            "text": f"A remarkably intricate and exceptionally verbose test sentence number {i} for evaluation.",
            "label": [1],
            "split": ["test"],
        })

    return records


def dummy_probing_record_handler(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Assigns a unique label to each text for the probing task.

    Args:
        example: A dataset record with "text", "label", and "split" fields.

    Returns:
        The example with a "steb_unique_label" field added.
    """
    text = example.get("text", "")
    text_id = hashlib.md5(text.encode("utf-8")).hexdigest()

    metadata = {
        "text_id": text_id,
        "label": example.get("label"),
        "split": example.get("split"),
    }
    example["steb_unique_label"] = json.dumps(metadata)
    return example
