import os
import numpy as np
from typing import Dict, List, Any


def load_fisher(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load Fisher speaker attribution dataset from .npy trial file.
    
    Expected structure:
    - data_dir can be a directory containing a .npy file, or
    - data_dir can be the path to a specific .npy file
    
    The .npy file contains an array of trial dictionaries with format:
    [
        {
            'label': 1 or 0,  # 1 = same speaker (positive), 0 = different speaker (negative)
            'call 1': ['utterance1', 'utterance2', ...],  # List of utterance strings
            'call 2': ['utterance1', 'utterance2', ...]   # List of utterance strings
        },
        ...
    ]
    
    Args:
        data_dir: Path to directory containing .npy file, or path to .npy file itself if there are multiple .npy files
        
    Returns:
        List of records with 'text' (list of utterances, preserved as separate strings) 
        and 'label' (trial_N_true or trial_N_false format to preserve trial identity)
        Each trial produces two records: one for call 1, one for call 2, both with same label
    """

    # Check if data_dir is a file or directory
    if os.path.isfile(data_dir) and data_dir.endswith('.npy'):
        npy_path = data_dir
    elif os.path.isdir(data_dir):
        # Look for .npy files in the directory
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")
        elif len(npy_files) > 1:
            raise ValueError(f"Multiple .npy files found in {data_dir}. Please specify the exact file path or ensure only one .npy file exists.")
        npy_path = os.path.join(data_dir, npy_files[0])
    else:
        raise FileNotFoundError(f"Data directory or file not found: {data_dir}")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Fisher trial file not found at {npy_path}")
    
    # Load the numpy array
    with open(npy_path, 'rb') as f:
        trials = np.load(f, allow_pickle=True)
    
    # Convert numpy array to list if needed
    if isinstance(trials, np.ndarray):
        trials = trials.tolist()
    
    records = []
    for trial_idx, trial in enumerate(trials):
        # Extract trial data
        label = trial.get('label')
        call1_utts = trial.get('call 1', [])
        call2_utts = trial.get('call 2', [])
        
        # Skip if missing required fields
        if label is None or not call1_utts or not call2_utts:
            continue
        
        # Convert utterances to strings if they're not already
        # Handle both list of strings and numpy arrays
        if isinstance(call1_utts, np.ndarray):
            call1_utts = call1_utts.tolist()
        if isinstance(call2_utts, np.ndarray):
            call2_utts = call2_utts.tolist()
        
        # Filter out non-string items, keeping utterances as separate strings in a list
        call1_text = [str(utt) for utt in call1_utts if isinstance(utt, (str, int, float)) and str(utt).strip()]
        call2_text = [str(utt) for utt in call2_utts if isinstance(utt, (str, int, float)) and str(utt).strip()]
        
        # Skip if either call has no valid text
        if not call1_text or not call2_text:
            continue
        
        # Create label string: trial_N_true or trial_N_false to preserve trial identity
        # true = same speaker (positive), false = different speaker (negative)
        if label == 1:
            label_str = f"trial_{trial_idx}_true"
        elif label == 0:
            label_str = f"trial_{trial_idx}_false"
        else:
            # Skip if label is not true or false
            continue
        
        # Create two records: one for call 1, one for call 2, both with same label
        # This matches the pattern used in pan15_dataset loader
        records.append({"text": call1_text, "label": label_str})
        records.append({"text": call2_text, "label": label_str})
    
    return records
