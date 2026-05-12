import os
from typing import Any, Dict, List

from datasets import load_dataset

from steb.steb_datasets.SynthSTEL.loader import synthstel_record_handler
from steb.utils import CACHE_DIR


SYNTHSTEL_FEATURE_MAP: Dict[str, str] = {
    "active_voice": "usage of active voice",
    "affective_process": "usage of words indicating affective process",
    "affective_processes": "usage of words indicating affective processes",
    "articles": "usage of articles",
    "certain_tone": "usage of certain tone (lack of uncertain words/phrases like 'i think', 'might', 'seems')",
    "cognitive_processes": "usage of words indicating cognitive processes",
    "complex_sentence_structure": "complex sentence structure",
    "conjunctions": "usage of conjunctions",
    "contractions": "usage of contractions",
    "emojis": "usage of emojis",
    "fluency_in_sentence_construction": "fluency in sentence construction",
    "formal_tone": "usage of formal tone",
    "frequent_common_verbs": "frequent usage of common verbs",
    "frequent_determiners": "frequent usage of determiners",
    "frequent_function_words": "frequent usage of function words",
    "frequent_punctuation": "frequent usage of punctuation",
    "humor": "incorporation of humor",
    "long_words": "usage of long words",
    "metaphors": "usage of metaphors",
    "misspelled_words": "presence of misspelled words",
    "nominalizations": "usage of nominalizations",
    "numerical_digits": "usage of numerical digits",
    "numerical_substitution": "usage of numerical substitution (ex. hello -> h3llo type slang)",
    "offensive_language": "usage of offensive language",
    "only_lowercase": "usage of only lowercase letters (all lowercase style)",
    "only_uppercase": "usage of only uppercase letters (all uppercase style)",
    "personal_pronouns": "usage of personal pronouns",
    "polite_tone": "usage of polite tone",
    "positive_sentiment": "positive sentiment expression",
    "prepositions": "usage of prepositions",
    "present_focused_tense": "usage of present-focused tense and words",
    "present_tense": "usage of present tense and present-focused words",
    "pronouns": "usage of pronouns",
    "sarcasm": "usage of sarcasm",
    "self_focused_language": "usage of self-focused language",
    "self_focused_perspective": "usage of self-focused perspective or words",
    "text_emojis": "usage of text emojis (ex. :-d type emoticons)",
    "uppercase_letters": "usage of uppercase letters",
}


def load_synthstel(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load the SynthSTEL HuggingFace dataset filtered to a single feature.

    The trailing component of `data_dir` names a short feature key in
    SYNTHSTEL_FEATURE_MAP, which is mapped to the full `feature_clean`
    label as it appears on HuggingFace. This mirrors the pastel split
    pattern: each per-feature dataset directory points to this loader
    with `"data_dir": "synthstel/<short_key>"`.

    Args:
        data_dir: Path whose basename is a key of SYNTHSTEL_FEATURE_MAP.

    Returns:
        List of {"text": [most_style, least_style], "label": str} records
        whose label equals the mapped full feature string.
    """
    short_name = os.path.basename(data_dir.rstrip(os.sep))
    if short_name not in SYNTHSTEL_FEATURE_MAP:
        raise ValueError(
            f"Unknown SynthSTEL short name {short_name!r}; "
            f"expected one of {sorted(SYNTHSTEL_FEATURE_MAP)}"
        )
    target_feature = SYNTHSTEL_FEATURE_MAP[short_name]

    ds = load_dataset("StyleDistance/synthstel", split="train", cache_dir=CACHE_DIR)

    records: List[Dict[str, Any]] = []
    for example in ds:
        record = synthstel_record_handler(example)
        if record is None:
            continue
        if record["label"] != target_feature:
            continue
        records.append(record)

    return records
