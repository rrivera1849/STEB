"""
Loader for StylePTB (Lyu et al., NAACL 2021).

Source: https://github.com/lvyiwei1/StylePTB
Paper: https://aclanthology.org/2021.naacl-main.171.pdf

StylePTB is a fine-grained controllable text style-transfer benchmark built
on top of the Penn Treebank. It provides parallel ``(source, target)``
sentence pairs across 21 transformation types (voice, tense, emphasis,
removal/addition, synonym/antonym substitution, ...). Each pair shares
content but differs along one "stylistic" dimension.

Subtypes shipped: 15 of 21
==========================
Six of the 21 transformations are excluded:

* ``NSR``/``ASR``/``VSR`` (noun/adjective/verb synonym replacement) --
  single-word lexical swaps with no consistent stylistic axis
* ``NAR``/``AAR``/``VAR`` (noun/adjective/verb antonym replacement) --
  these change *meaning* (e.g. "happy" -> "sad"), violating the
  same-content assumption that makes order-alignment interpretable here.

The remaining 15 are:

===== =============================== ===============================================
Code  Subtype label                   Description
===== =============================== ===============================================
TFU   to_future                       Tense -> future
TPA   to_past                         Tense -> past
TPR   to_present                      Tense -> present
ATP   active_to_passive               Active -> passive voice
PTA   passive_to_active               Passive -> active voice
PFB   pp_front_to_back                Move PP from front to back of sentence
PBF   pp_back_to_front                Move PP from back to front of sentence
ARR   adjective_adverb_removal        Strip adjectives and adverbs
SBR   substatement_removal            Strip subordinate clauses
PPR   pp_removal                      Strip prepositional phrases
IAD   information_addition            Add modifiers / extra information
AEM   adjective_emphasis              Add adjective intensifiers
VEM   verb_emphasis                   Add verb intensifiers
LFS   least_frequent_synonym          Replace word with its rarest synonym
MFS   most_frequent_synonym           Replace word with its commonest synonym
===== =============================== ===============================================

Splits
======
StylePTB ships train/dev/test splits per subtype, but they are produced
by the upstream ``single_transform_checkout.py`` script from a single
master file. STEB has no training step, so we read the master file
``fulldata.h16`` directly and use every parallel pair (no split).

Caveat: PTB tokenisation
========================
The text is the original Penn Treebank tokenisation: lowercased, with
contractions split at the apostrophe (``do n't``, ``wo n't``, ``it 's``)
and punctuation treated as separate tokens. We do not
detokenize.
"""

import os
from typing import Any, Dict, List


# Mapping from StylePTB 3-letter codes to descriptive labels.
# Codes deliberately excluded (see module docstring): NSR, ASR, VSR,
# NAR, AAR, VAR.
_KEPT_CODES: Dict[str, str] = {
    "TFU": "to_future",
    "TPA": "to_past",
    "TPR": "to_present",
    "ATP": "active_to_passive",
    "PTA": "passive_to_active",
    "PFB": "pp_front_to_back",
    "PBF": "pp_back_to_front",
    "ARR": "adjective_adverb_removal",
    "SBR": "substatement_removal",
    "PPR": "pp_removal",
    "IAD": "information_addition",
    "AEM": "adjective_emphasis",
    "VEM": "verb_emphasis",
    "LFS": "least_frequent_synonym",
    "MFS": "most_frequent_synonym",
}


def load_styleptb_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load the StylePTB single-transform dataset for order alignment.

    Reads ``fulldata.h16`` from ``data_dir``. The file is a flat sequence
    of triplets, one per parallel pair:

        <3-letter transformation code>
        <source sentence>
        <target sentence>

    Pairs whose code is not in the 15 kept subtypes are skipped
    (see module docstring).

    Args:
        data_dir: Path to the directory holding ``fulldata.h16``,
            typically ``raw_datasets/styleptb``.

    Returns:
        A list of records, each ``{"text": [target, source], "label":
        "<subtype_label>"}``. ``label`` is the descriptive subtype name
        (e.g. ``"active_to_passive"``); records sharing a label are
        compared pairwise by the order-alignment task.
    """
    fulldata_path = os.path.join(data_dir, "fulldata.h16")
    if not os.path.isfile(fulldata_path):
        raise FileNotFoundError(f"fulldata.h16 not found in: {data_dir}")

    with open(fulldata_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    records: List[Dict[str, Any]] = []
    n = len(lines)
    for i in range(n - 2):
        code = lines[i][:3]
        if code not in _KEPT_CODES:
            continue
        source = lines[i + 1]
        target = lines[i + 2]
        if not source or not target:
            continue
        records.append({
            "text": [target, source],
            "label": _KEPT_CODES[code],
        })

    return records
