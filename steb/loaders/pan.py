import json
import os
from typing import Any, Dict, List

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


def load_pan13_dataset(
    data_dir: str,
) -> list[dict[str, str]]:
    """
    Loads a PAN13 authorship verification dataset for a single language.

    The *data_dir* argument encodes the corpus path and the target
    language, e.g. ``…/pan13-test/EN``.  The last component is used as
    a prefix filter (``EN``, ``GR``, or ``SP``) and the parent
    directory is expected to contain ``truth.txt`` and the problem
    subdirectories.

    PAN13 problems may contain multiple known-author documents
    (``known01.txt`` … ``knownNN.txt``).  All known documents are
    concatenated (separated by double newlines) into a single text
    that forms one side of the verification pair; ``unknown.txt``
    forms the other.

    Args:
        data_dir: Path whose last component is the language prefix
                  (e.g. ``…/pan13-corpus/EN``).

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs (known, unknown) per problem.
    """
    corpus_dir = os.path.dirname(data_dir)
    language_prefix = os.path.basename(data_dir)

    truth_path = os.path.join(corpus_dir, "truth.txt")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth.txt not found in {corpus_dir}")

    problem_labels: dict[str, int] = {}
    with open(truth_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pid = parts[0]
                label_char = parts[1]
                problem_labels[pid] = 1 if label_char == "Y" else 0

    sorted_pids = sorted(
        pid for pid in problem_labels
        if pid.startswith(language_prefix)
    )

    samples: list[dict[str, str]] = []
    for pid in sorted_pids:
        problem_dir = os.path.join(corpus_dir, pid)
        if not os.path.exists(problem_dir):
            continue

        unknown_path = os.path.join(problem_dir, "unknown.txt")
        if not os.path.exists(unknown_path):
            continue

        known_files = sorted(
            f for f in os.listdir(problem_dir)
            if f.startswith("known") and f.endswith(".txt")
        )
        if not known_files:
            continue

        try:
            known_texts = []
            for kf in known_files:
                with open(os.path.join(problem_dir, kf), "r", encoding="utf-8", errors="replace") as f:
                    known_texts.append(f.read())
            text_a = "\n\n".join(known_texts)

            with open(unknown_path, "r", encoding="utf-8", errors="replace") as f:
                text_b = f.read()
        except Exception:
            continue

        if problem_labels[pid] == 1:
            label_str = f"trial_{pid}_true"
        else:
            label_str = f"trial_{pid}_false"

        samples.append({"text": text_a, "label": label_str})
        samples.append({"text": text_b, "label": label_str})

    return samples


def load_pan_jsonl_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a PAN authorship verification dataset stored as JSONL files.

    Works with PAN20, PAN21, and any future edition that uses the same
    format.  The directory must contain exactly two ``.jsonl`` files:
    one whose name ends with ``-truth.jsonl`` (the labels) and one
    that does not (the data).

    Each data record has an ``id``, ``fandoms``, and ``pair`` (a list
    of two texts).  Each truth record has an ``id`` and ``same``
    (boolean).

    The dataset is expected to have been subsampled at download time
    (see ``download_datasets.sh``).

    Args:
        data_dir: Path to the directory containing the JSONL files.

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs (text_a, text_b) per problem.
    """
    jsonl_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".jsonl")
    )
    truth_files = [f for f in jsonl_files if f.endswith("-truth.jsonl")]
    data_files = [f for f in jsonl_files if not f.endswith("-truth.jsonl")]

    if len(truth_files) != 1 or len(data_files) != 1:
        raise FileNotFoundError(
            f"Expected one data and one truth JSONL file in {data_dir}, "
            f"found {jsonl_files}"
        )

    data_path = os.path.join(data_dir, data_files[0])
    truth_path = os.path.join(data_dir, truth_files[0])

    with open(truth_path, "r") as f:
        truth_by_id = {
            t["id"]: t["same"]
            for t in (json.loads(line) for line in f if line.strip())
        }

    samples: List[Dict[str, Any]] = []
    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record["id"]
            pair = record["pair"]

            if pid not in truth_by_id or len(pair) != 2:
                continue

            same = truth_by_id[pid]
            label_str = f"trial_{pid}_true" if same else f"trial_{pid}_false"

            samples.append({"text": pair[0], "label": label_str})
            samples.append({"text": pair[1], "label": label_str})

    return samples
