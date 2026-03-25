import json
import os
from typing import Any, Dict, List


def load_radiotalk_pairs(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load RadioTalk speaker pairs from JSON file for pre_defined_pair_classification.
    
    Expected format: JSON array of objects with:
    {
        "label": 1 or 0,  # 1 = same speaker (positive), 0 = different speaker (negative)
        "speaker 1": ["utterance1", "utterance2", ...],
        "speaker 2": ["utterance1", "utterance2", ...]
    }
    
    Args:
        data_dir: Path to the directory containing radiotalk_pairs.json
        
    Returns:
        List of records with 'text' (list of utterances) and 'label' (trial_N_true or trial_N_false)
        Each pair produces two records: one for speaker 1, one for speaker 2, both with same label
        True label is 1 (same speaker), False label is 0 (different speaker)
    """
    json_path = os.path.join(data_dir, "radiotalk_pairs.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"RadioTalk pairs file not found at {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    if not isinstance(pairs, list):
        raise ValueError(f"Expected JSON array, got {type(pairs)}")

    records = []
    for trial_idx, pair in enumerate(pairs):
        label = pair.get('label')
        speaker1_utts = pair.get('speaker 1', [])
        speaker2_utts = pair.get('speaker 2', [])
        
        if label is None or not speaker1_utts or not speaker2_utts:
            continue
        
        speaker1_text = [str(utt) for utt in speaker1_utts if isinstance(utt, (str, int, float)) and str(utt).strip()]
        speaker2_text = [str(utt) for utt in speaker2_utts if isinstance(utt, (str, int, float)) and str(utt).strip()]
        
        if not speaker1_text or not speaker2_text:
            continue

        if label == 1:
            label_str = f"trial_{trial_idx}_true"
        elif label == 0:
            label_str = f"trial_{trial_idx}_false"
        else:
            continue
        
        records.append({"text": speaker1_text, "label": label_str})
        records.append({"text": speaker2_text, "label": label_str})

    return records
