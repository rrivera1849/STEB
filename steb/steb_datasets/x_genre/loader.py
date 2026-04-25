from typing import Any, Dict, List

from datasets import load_dataset


HF_PATH = "TajaKuzmanPungersek/X-GENRE-text-genre-dataset"
HF_CONFIGS = ["train", "test", "dev"]
KEEP_SOURCES = {"CORE", "FTD"}


def load_x_genre_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load the X-GENRE genre-classification dataset, restricted to the CORE
    and FTD source-subsets and merged across the train, test, and dev
    HuggingFace configs.

    The HuggingFace dataset exposes ``train`` / ``test`` / ``dev`` as named
    configs (not splits); each config is loaded as a DatasetDict and every
    split inside it is consumed. Every row has a ``dataset`` field
    (``CORE`` / ``FTD`` / ``GINCO``); GINCO rows are Slovenian and are
    dropped. The genre class is read from the ``labels`` field and
    re-emitted under STEB's standard ``label`` key.

    Args:
        data_dir: Unused. Required by the STEB custom-loader contract;
            the HuggingFace dataset id is hardcoded and HF handles caching.

    Returns:
        A list of records with ``text`` and ``label`` fields, one per
        retained CORE/FTD row across all three configs.
    """
    records: List[Dict[str, Any]] = []
    for config_name in HF_CONFIGS:
        ds_dict = load_dataset(HF_PATH, name=config_name)
        rows = [row for split_rows in ds_dict.values() for row in split_rows]
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
