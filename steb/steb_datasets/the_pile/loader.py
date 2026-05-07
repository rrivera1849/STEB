from typing import Any, Dict, List

from datasets import get_dataset_config_names, load_dataset


HF_PATH = "ArmelR/the-pile-splitted"
SAMPLES_PER_CATEGORY = 1000
SKIP_CONFIGS = {"all"}
MAX_TEXT_LENGTH = 1_000_000


def load_the_pile_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load samples from each category of the Pile via the
    ``ArmelR/the-pile-splitted`` HuggingFace dataset.

    The HuggingFace dataset exposes one config per Pile category
    (Pile-CC, ArXiv, Github, Books3, ...) plus an ``all`` config that
    aggregates them. For each per-category config we stream the train
    split and collect the first ``SAMPLES_PER_CATEGORY`` rows whose
    text is non-empty and shorter than ``MAX_TEXT_LENGTH`` characters,
    using the config name as the label. Rows at or above the length
    cap are skipped to avoid downstream spaCy E088 errors during
    sentence splitting. The ``all`` config is skipped to avoid
    duplicating samples under a separate label.

    Args:
        data_dir: Unused. Required by the STEB custom-loader contract;
            the HuggingFace dataset id is hardcoded and HF handles caching.

    Returns:
        A list of records with ``text`` and ``label`` fields, one per
        sampled row across all Pile categories.
    """
    records: List[Dict[str, Any]] = []
    for config_name in sorted(get_dataset_config_names(HF_PATH)):
        if config_name in SKIP_CONFIGS:
            continue
        ds = load_dataset(HF_PATH, name=config_name, split="train", streaming=True)
        collected = 0
        for row in ds:
            if collected >= SAMPLES_PER_CATEGORY:
                break
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if len(text) >= MAX_TEXT_LENGTH:
                continue
            records.append({"text": text, "label": config_name})
            collected += 1
    return records
