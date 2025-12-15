import os
import json

def load_pan15_dataset(data_dir):
    """
    Loads the PAN15 authorship verification dataset.
    
    Expected structure:
    data_dir/
      truth.txt
      EN001/
        known01.txt
        unknown.txt
      EN002/
      ...
    """
    truth_path = os.path.join(data_dir, "truth.txt")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth.txt not found in {data_dir}")

    # Read truth file
    # Format: EN001 Y
    problem_labels = {}
    with open(truth_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pid = parts[0]
                label_char = parts[1]
                # Y -> 1 (Same), N -> 0 (Different)
                problem_labels[pid] = 1 if label_char == 'Y' else 0

    # Iterate over problems
    # To rely on sequential grouping, we must yield pairs strictly together.
    # We sort keys to ensure deterministic order if that matters, but crucial is (A,B) order.
    
    sorted_pids = sorted(problem_labels.keys())
    
    samples = []
    for pid in sorted_pids:
        problem_dir = os.path.join(data_dir, pid)
        if not os.path.exists(problem_dir):
            continue
            
        known_path = os.path.join(problem_dir, "known01.txt")
        unknown_path = os.path.join(problem_dir, "unknown.txt")
        
        if not os.path.exists(known_path) or not os.path.exists(unknown_path):
            continue
            
        try:
            with open(known_path, "r", encoding="utf-8", errors="replace") as f:
                text_a = f.read()
            with open(unknown_path, "r", encoding="utf-8", errors="replace") as f:
                text_b = f.read()
        except Exception:
            # Skip if read error
            continue
            
        if problem_labels[pid] == 1:
            label_str = f"trial_{pid}_true"
        else:
            label_str = f"trial_{pid}_false"
        
        # Yield Text A then Text B
        samples.append({"text": text_a, "label": label_str})
        samples.append({"text": text_b, "label": label_str})
        
    return samples
