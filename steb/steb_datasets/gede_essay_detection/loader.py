import json
import os
from typing import Dict, List, Any


def load_gede(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load GEDE dataset from JSON file.
    
    Args:
        data_dir: Path to the directory containing gede_essays.json
        
    Returns:
        List of records with 'text' (from 'answer' field) and 'label' (from 'text_author' field)
    """
    json_path = os.path.join(data_dir, "gede_essays.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"GEDE dataset file not found at {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = []
    for entry in data:
        answer = entry.get("answer")
        text_author = entry.get("text_author")
        
        # Skip entries without required fields
        if not answer or not text_author:
            continue
        
        # Convert answer to string if it's not already
        if not isinstance(answer, str):
            continue
        
        records.append({
            "text": answer,
            "label": text_author
        })
    
    return records

