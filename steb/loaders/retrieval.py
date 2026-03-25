import glob
import json
import os
from typing import Any, Dict, List, Optional


def default_retrieval_loader(data_dir: str) -> List[Dict[str, Any]]:
    """
    Iterates over all files in the data directory and yields examples.
    Assumes files are JSONL.
    """
    files = glob.glob(os.path.join(data_dir, "*"))
    examples = []
    for file_path in files:
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return examples


def default_retrieval_record_handler(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts text and label, and appends _query or _target suffix to label based on is_query field.

    NOTE: We're assuming that we have dictionaries with the following fields: `text`, `label`, and `is_query`
    """
    text = example.get("text")
    label = example.get("label")
    is_query = example.get("is_query", False)

    label = str(label)
    suffix = "_query" if is_query else "_target"

    return {"text": text, "label": f"{label}{suffix}"}
