"""
Loader for eWAVE (Kortmann, Lunkenheimer & Ehret 2020).

Source: https://zenodo.org/records/17433568
License: CC-BY 3.0
Cite: Kortmann, Bernd & Lunkenheimer, Kerstin & Ehret, Katharina (eds.) 2020.
The Electronic World Atlas of Varieties of English.

eWAVE catalogues morphosyntactic features across world Englishes (regional
L1, high-contact L1, indigenized L2, English-based pidgins, English-based
creoles). The CLDF release ships ~4,200 example sentences attached to a
specific variety; we use those sentences as records and the variety's full
name as the label.

Join
====
``cldf/examples.csv.Language_ID`` -> ``cldf/languages.csv.ID`` -> variety
``Name`` (e.g. ``"Scottish English"``, ``"Urban African American Vernacular
English"``).

Caveat: uneven coverage
=======================
"""

import csv
import os
from collections import Counter
from typing import Any, Dict, List


_MIN_EXAMPLES_PER_VARIETY: int = 2


def load_eWAVE_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load eWAVE example sentences labeled by variety of English.

    Reads ``cldf/examples.csv`` and ``cldf/languages.csv`` from ``data_dir``,
    joins them on ``Language_ID`` -> ``ID``, and emits one record per example
    sentence. Rows with empty ``Primary_Text`` or an unknown ``Language_ID``
    are skipped. Varieties with fewer than ``_MIN_EXAMPLES_PER_VARIETY``
    surviving records are dropped.

    Args:
        data_dir: Path to the directory holding the unpacked CLDF release,
            typically ``raw_datasets/eWAVE`` (must contain
            ``cldf/examples.csv`` and ``cldf/languages.csv``).

    Returns:
        A list of records ``{"text": <example sentence>, "label": <variety
        name>}``, one per surviving example sentence.
    """
    examples_path = os.path.join(data_dir, "cldf", "examples.csv")
    languages_path = os.path.join(data_dir, "cldf", "languages.csv")
    if not os.path.isfile(examples_path):
        raise FileNotFoundError(f"eWAVE examples.csv not found: {examples_path}")
    if not os.path.isfile(languages_path):
        raise FileNotFoundError(f"eWAVE languages.csv not found: {languages_path}")

    with open(languages_path, "r", encoding="utf-8", newline="") as f:
        variety_name_by_id: Dict[str, str] = {
            row["ID"]: row["Name"] for row in csv.DictReader(f)
        }

    records: List[Dict[str, Any]] = []
    with open(examples_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("Primary_Text") or "").strip()
            if not text:
                continue
            variety = variety_name_by_id.get(row.get("Language_ID", ""))
            if variety is None:
                continue
            records.append({"text": text, "label": variety})

    counts = Counter(r["label"] for r in records)
    return [r for r in records if counts[r["label"]] >= _MIN_EXAMPLES_PER_VARIETY]
