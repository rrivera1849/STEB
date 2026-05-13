import os
from typing import Any, Dict, List

from datasets import load_dataset

from steb.steb_datasets.SynthSTEL.loader import synthstel_record_handler
from steb.utils import CACHE_DIR


# SynthSTEL `feature` (raw contrastive) values that we treat as
# register (formality, politeness, emotional tone, sarcasm,
# humor, certain-tone, offensiveness, positive sentiment, complex-vs-simple
# as the parallel to STEL simplicity). Every other SynthSTEL feature is
# treated as a single manipulated linguistic feature.
#
# Strings match the upstream `feature` column (40 distinct values in the
# StyleDistance/synthstel HF dataset) rather than `feature_clean` (38
# values, which collapses three self-focused contrasts into one label).
SYNTHSTEL_REGISTER_FEATURES: frozenset = frozenset({
    "certain / uncertain",
    "complex / simple",
    "formal / informal",
    "offensive / non-offensive",
    "polite / impolite",
    "positive / negative",
    "with humor / without humor",
    "with sarcasm / without sarcasm",
})

SYNTHSTEL_GROUPS: frozenset = frozenset({"register", "feature"})


def load_synthstel(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load SynthSTEL from HuggingFace, filtered to one of two subset groups.

    The trailing component of `data_dir` selects the subset group:

      - ``register`` keeps records whose ``feature`` label is in
        :data:`SYNTHSTEL_REGISTER_FEATURES` (formality, politeness, emotional
        tone, sarcasm, humor, certain-tone, offensiveness, positive sentiment,
        complex sentence structure).
      - ``feature`` keeps every other SynthSTEL feature (surface /
        linguistic categories such as contractions, emojis, function-word
        usage, pronoun usage, etc.).

    Args:
        data_dir: Path whose basename is a member of :data:`SYNTHSTEL_GROUPS`.

    Returns:
        List of ``{"text": [most_style, least_style], "label": str}`` records
        whose label is in (or, for ``feature``, outside of) the register set.
    """
    group = os.path.basename(data_dir.rstrip(os.sep))
    if group not in SYNTHSTEL_GROUPS:
        raise ValueError(
            f"Unknown SynthSTEL subset group {group!r}; "
            f"expected one of {sorted(SYNTHSTEL_GROUPS)}"
        )

    # Load both upstream splits and concatenate. The train/test split in the
    # StyleDistance/synthstel release reflects the original embedding-training
    # protocol; STEB uses SynthSTEL only as an evaluation benchmark for
    # order_alignment, so all 40 features x (90 train + 10 test) = 4000 pairs
    # are equally valid evaluation data and combining them reduces per-label
    # variance (100 rows per feature instead of 90).
    ds = load_dataset("StyleDistance/synthstel", split="train+test", cache_dir=CACHE_DIR)

    records: List[Dict[str, Any]] = []
    for example in ds:
        record = synthstel_record_handler(example)
        if record is None:
            continue
        is_register = record["label"] in SYNTHSTEL_REGISTER_FEATURES
        if group == "register" and not is_register:
            continue
        if group == "feature" and is_register:
            continue
        records.append(record)

    return records
