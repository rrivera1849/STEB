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

    Args:
        example: A dataset record with "text", "label", and "split" fields.

    Returns:
        The example with a "steb_unique_label" field added containing serialized metadata.
    """
    text = example.get("text", "")
    text_id = hashlib.md5(text.encode("utf-8")).hexdigest()

    metadata = {
        "text_id": text_id,
        "label": example.get("label"),
        "split": example.get("split")
    }
    example["steb_unique_label"] = json.dumps(metadata)
    return example
