import hashlib
import json
from typing import Any, Dict, List, Optional


def load_probing_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads a probing dataset from a specific JSONL file.

    Args:
        data_path: Path to the JSONL file.

    Returns:
        A list of parsed JSON records.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def probing_record_handler(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Assigns a unique label to each text so DatasetLoader treats them as separate samples.

    Reconstructs feature labels and splits from the individual per-feature
    fields (e.g. label_n_adj, split_n_adj) rather than relying on the
    pre-built positional lists, so that feature names are preserved explicitly.

    Args:
        example: A dataset record with per-feature label_* and split_* fields.

    Returns:
        The example with a "steb_unique_label" field containing serialized
        metadata with feature_names, labels, and splits keyed by feature name.
    """
    text = example.get("text")
    text_id = hashlib.md5(text.encode("utf-8")).hexdigest()

    feature_names = [
        k[len("label_"):] for k in example
        if k.startswith("label_")
    ]

    labels = {name: example[f"label_{name}"] for name in feature_names}
    splits = {name: example[f"split_{name}"] for name in feature_names}

    metadata = {
        "text_id": text_id,
        "feature_names": feature_names,
        "labels": labels,
        "splits": splits,
    }
    example["steb_unique_label"] = json.dumps(metadata)
    return example
