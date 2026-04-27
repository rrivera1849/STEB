import json
import os
from typing import Any, Dict, List


def load_politeness_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load a Stanford politeness corpus exported via ConvoKit.

    Reads ``utterances.jsonl`` from ``data_dir``. Each line is a JSON
    record whose ``meta`` dict carries the politeness annotation under
    the ``Binary`` key, where 1 = polite, 0 = neutral, -1 = impolite
    (per the ConvoKit documentation).

    Utterances with empty text or with a missing / non-integer ``Binary``
    annotation are skipped.

    Args:
        data_dir: Path to the unpacked ConvoKit corpus directory
            (e.g. ``raw_datasets/wikipedia-politeness-corpus``), as
            produced by the curl/unzip blocks in
            ``download_datasets.sh``.

    Returns:
        A list of records ``{"text": str, "label": str}`` with labels in
        ``{impolite, neutral, polite}``.
    """
    utterances_path = os.path.join(data_dir, "utterances.jsonl")
    if not os.path.isfile(utterances_path):
        raise FileNotFoundError(f"utterances.jsonl not found in: {data_dir}")

    records: List[Dict[str, Any]] = []
    with open(utterances_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterance = json.loads(line)

            text = utterance.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            binary = utterance.get("meta", {}).get("Binary")
            if not isinstance(binary, int):
                continue
            if binary not in (-1, 0, 1):
                continue
            label = {
                0: "neutral",
                1: "polite",
                -1: "impolite",
            }[binary]

            records.append({"text": text, "label": label})

        if len(records) == 0:
            raise ValueError(f"No valid records found in: {utterances_path}")

    return records
