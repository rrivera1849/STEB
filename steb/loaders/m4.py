import json
import os
from typing import Any, Dict, List


# Domains whose files have list-valued human_text / machine_text fields.
_LIST_VALUED_DOMAINS = {"peerread"}


def _read_jsonl(
    path: str,
) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file and returns a list of parsed records.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries, one per line.
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_m4(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a single-domain slice of the M4 machine text detection dataset.

    The *data_dir* argument encodes both the dataset root and the target
    domain, e.g. ``/path/to/raw_datasets/M4/arxiv``.  The last path
    component is treated as the domain prefix; data files are read from a
    sibling ``data/`` directory by matching ``{domain}_*.jsonl``.

    Only files that contain ``human_text`` and ``machine_text`` fields are
    used (this excludes the bloomz variants whose schema differs).  Human
    texts are deduplicated so that each unique text appears once with the
    label ``"human"``.

    For domains with list-valued text fields (e.g. peerread, where each
    record contains multiple reviews), the lists are flattened into
    individual records.

    Args:
        data_dir: Path whose last component is the domain prefix
                  (e.g. ``…/M4/arxiv``).

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    base_dir = os.path.dirname(data_dir)
    domain = os.path.basename(data_dir)
    files_dir = os.path.join(base_dir, "data")

    domain_files = sorted(
        f for f in os.listdir(files_dir)
        if f.endswith(".jsonl") and f.startswith(domain + "_")
    )

    human_texts_seen: set = set()
    machine_texts_seen: set = set()
    records: List[Dict[str, Any]] = []

    for fname in domain_files:
        rows = _read_jsonl(os.path.join(files_dir, fname))
        if not rows or "human_text" not in rows[0]:
            continue

        model = rows[0].get("model", fname) # RRS - One model per file
        is_list_domain = domain in _LIST_VALUED_DOMAINS

        for row in rows:
            human_raw = row.get("human_text")
            machine_raw = row.get("machine_text")
            if human_raw is None or machine_raw is None:
                continue

            if is_list_domain:
                human_items = human_raw if isinstance(human_raw, list) else [human_raw]
                machine_items = machine_raw if isinstance(machine_raw, list) else [machine_raw]
            else:
                human_items = [human_raw]
                machine_items = [machine_raw]

            for text in human_items:
                if not text or not isinstance(text, str):
                    continue
                if text not in human_texts_seen:
                    human_texts_seen.add(text)
                    records.append({"text": text, "label": "human"})

            for text in machine_items:
                if not text or not isinstance(text, str):
                    continue
                if text not in machine_texts_seen:
                    machine_texts_seen.add(text)
                    records.append({"text": text, "label": model})

    return records
