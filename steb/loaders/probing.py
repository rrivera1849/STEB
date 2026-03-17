import os
import json
import hashlib
from typing import Dict, Any, Optional

def load_probing_dataset(data_path: str):
    """
    Loads a probing dataset from a specific JSONL file.
    """
    
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def probing_record_handler(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ensure every single text yields a unique label mapping so they don't get combined by DatasetLoader
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
