from typing import Any, Dict, List

from steb.loaders.probing import probing_record_handler  # noqa: F401


def load_dummy_probing_dataset(_path: str) -> List[Dict[str, Any]]:
    """
    Provides a minimal synthetic dataset for probing task tests.

    Generates records with a single probing feature (sentence_length,
    binary classification). Each record has text and per-feature
    label_*/split_* fields following the probing dataset format.

    Args:
        _path: Unused, kept for loader interface compatibility.

    Returns:
        A list of records with 'text' and per-feature label/split fields.
    """
    records = []

    # Training set: 40 samples per class
    for i in range(40):
        records.append({
            "text": f"This is a short simple sentence number {i}.",
            "label_sentence_length": 0,
            "split_sentence_length": "train",
        })
        records.append({
            "text": f"The extraordinarily complex and multifaceted nature of this particular sentence number {i} is remarkable.",
            "label_sentence_length": 1,
            "split_sentence_length": "train",
        })

    # Validation set: 10 samples per class
    for i in range(10):
        records.append({
            "text": f"A brief sentence for validation {i}.",
            "label_sentence_length": 0,
            "split_sentence_length": "val",
        })
        records.append({
            "text": f"An exceedingly elaborate and thoroughly detailed validation sentence number {i} indeed.",
            "label_sentence_length": 1,
            "split_sentence_length": "val",
        })

    # Test set: 10 samples per class
    for i in range(10):
        records.append({
            "text": f"Short test text {i}.",
            "label_sentence_length": 0,
            "split_sentence_length": "test",
        })
        records.append({
            "text": f"A remarkably intricate and exceptionally verbose test sentence number {i} for evaluation.",
            "label_sentence_length": 1,
            "split_sentence_length": "test",
        })

    return records
