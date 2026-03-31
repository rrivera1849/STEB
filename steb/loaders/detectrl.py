import json
from typing import Any, Dict, List


def _load_test_file(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Reads a DetectRL JSON test file.

    Args:
        data_dir: Path to the JSON file.

    Returns:
        A list of raw records.
    """
    with open(data_dir, "r") as f:
        return json.load(f)


def load_detectrl_multiclass(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a DetectRL test file with multi-class labels.

    Human records get the label ``"human"``.  LLM records get their
    ``llm_type`` value as the label (e.g. ``"ChatGPT"``,
    ``"Llama-2-70b"``), enabling multi-class clustering that
    distinguishes individual model styles.

    Args:
        data_dir: Path to the JSON test file.

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    raw = _load_test_file(data_dir)
    records: List[Dict[str, Any]] = []
    for row in raw:
        text = row.get("text")
        if not text or not isinstance(text, str):
            continue

        if row.get("label") == "human":
            label = "human"
        else:
            label = row.get("llm_type", "llm")

        records.append({"text": text, "label": label})
    return records


def load_detectrl_binary(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a DetectRL test file with binary labels (human vs llm).

    Uses the original ``label`` field directly, producing two clusters:
    ``"human"`` and ``"llm"``.  This is appropriate for attack datasets
    where the question is whether embeddings can still separate human
    from machine text after evasion techniques are applied.

    Args:
        data_dir: Path to the JSON test file.

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    raw = _load_test_file(data_dir)
    records: List[Dict[str, Any]] = []
    for row in raw:
        text = row.get("text")
        label = row.get("label")
        if not text or not isinstance(text, str):
            continue
        if label not in ("human", "llm"):
            continue

        records.append({"text": text, "label": label})
    return records
