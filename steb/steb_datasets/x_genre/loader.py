from typing import Any, Dict, List

from datasets import load_dataset


HF_PATH = "TajaKuzmanPungersek/X-GENRE-text-genre-dataset"
SPLITS = ["train", "test", "dev"]
KEEP_SOURCES = {"CORE", "FTD"}


def load_x_genre_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load the X-GENRE genre-classification dataset, restricted to the CORE
    and FTD source-subsets and merged across train, test, and dev splits.

    The HuggingFace dataset tags every row with a ``dataset`` field
    (``CORE`` / ``FTD`` / ``GINCO``); GINCO rows are Slovenian and are
    dropped here. The genre class is read from the ``labels`` field and
    re-emitted under STEB's standard ``label`` key.

    Args:
        data_dir: Unused. Required by the STEB custom-loader contract;
            the HuggingFace dataset id is hardcoded and HF handles caching.

    Returns:
        A list of records with ``text`` and ``label`` fields, one per
        retained CORE/FTD row across all three splits.
    """
    records: List[Dict[str, Any]] = []
    for split in SPLITS:
        rows = load_dataset(HF_PATH, split=split)
        for row in rows:
            if row.get("dataset") not in KEEP_SOURCES:
                continue
            text = row.get("text")
            label = row.get("labels")
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(label, str) or not label.strip():
                continue
            records.append({"text": text, "label": label})
    return records
