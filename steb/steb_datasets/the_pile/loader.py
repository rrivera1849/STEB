from itertools import islice
from typing import Any, Dict, List

from datasets import get_dataset_config_names, load_dataset


HF_PATH = "ArmelR/the-pile-splitted"
SAMPLES_PER_CATEGORY = 1000
SKIP_CONFIGS = {"all"}


def load_the_pile_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load samples from each category of the Pile via the
    ``ArmelR/the-pile-splitted`` HuggingFace dataset.

    The HuggingFace dataset exposes one config per Pile category
    (Pile-CC, ArXiv, Github, Books3, ...) plus an ``all`` config that
    aggregates them. For each per-category config we stream the train
    split and take the first ``SAMPLES_PER_CATEGORY`` rows, using the
    config name as the label. The ``all`` config is skipped to avoid
    duplicating samples under a separate label.

    Args:
        data_dir: Unused. Required by the STEB custom-loader contract;
            the HuggingFace dataset id is hardcoded and HF handles caching.

    Returns:
        A list of records with ``text`` and ``label`` fields, one per
        sampled row across all Pile categories.
    """
    records: List[Dict[str, Any]] = []
    for config_name in get_dataset_config_names(HF_PATH):
        if config_name in SKIP_CONFIGS:
            continue
        ds = load_dataset(HF_PATH, name=config_name, split="train", streaming=True)
        for row in islice(ds, SAMPLES_PER_CATEGORY):
            records.append({"text": row["text"], "label": config_name})
    return records
