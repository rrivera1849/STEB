import os
from typing import Any, Dict, List

from datasets import load_dataset


def _extract_label(
    src: str,
    domain: str,
) -> str | None:
    """
    Extracts a clustering label from a MAGE ``src`` field.

    Human records are labelled ``"human"``.  Machine records are labelled
    by model name only (the generation method — continuation, specified,
    topical — is stripped so that all outputs from the same model share a
    single label).

    Args:
        src: The ``src`` field value, e.g. ``"eli5_machine_continuation_flan_t5_base"``.
        domain: The domain prefix, e.g. ``"eli5"``.

    Returns:
        The label string, or ``None`` if the record should be skipped.
    """
    suffix = src[len(domain) + 1:]

    if suffix.startswith("human"):
        return "human"

    if suffix.startswith("machine_"):
        parts = suffix.split("_", 2)
        if len(parts) >= 3:
            return parts[2]

    return None


def load_mage(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a single-domain slice of the MAGE dataset (test split) from
    Hugging Face.

    The *data_dir* argument encodes the target domain as its last path
    component (e.g. ``…/MAGE/eli5``).  Records are filtered to that
    domain and labelled by model name for multi-class clustering.

    Args:
        data_dir: Path whose last component is the domain name.

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    domain = os.path.basename(data_dir)
    prefix = domain + "_"

    ds = load_dataset("yaful/MAGE", split="test")

    records: List[Dict[str, Any]] = []
    for row in ds:
        src = row.get("src", "")
        if not src.startswith(prefix):
            continue

        text = row.get("text")
        if not text or not isinstance(text, str):
            continue

        label = _extract_label(src, domain)
        if label is None:
            continue

        records.append({"text": text, "label": label})

    unique_labels = set(r["label"] for r in records)
    print(f"Unique labels: {unique_labels}")
    return records
