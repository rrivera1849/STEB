import glob
import json
import os
from typing import Any, Dict, List


def _load_pastel_stories(
    data_dir: str,
    attribute: str,
) -> List[Dict[str, Any]]:
    """
    Load PASTEL stories labelled by a single persona attribute.

    Each *.json file under data_dir is a single story object with keys
    'output.sentences' (list of 5 sentences rewritten in the annotator's
    persona) and 'persona' (dict with lowercase persona keys). The five
    output sentences are joined with a single space into one text string.

    Records whose persona attribute is missing, an empty string, or the
    upstream "Empty" sentinel are dropped. JSON files are gathered
    recursively, so data_dir may either be a single split directory
    (e.g. .../stories/test) or the parent directory containing all
    splits (e.g. .../stories), in which case train, valid, and test
    are all loaded.

    Args:
        data_dir: Path to a PASTEL stories directory containing per-story
            JSON files, optionally nested in train/valid/test subdirs.
        attribute: Persona key to use as the label. One of
            "age", "gender", "country", "ethnic", "education",
            "politics", "tod".

    Returns:
        List of {"text": str, "label": str} records, one per usable story.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"PASTEL stories directory not found: {data_dir}")

    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .json story files found under {data_dir}")

    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)

        sentences = obj.get("output.sentences", [])
        text = " ".join(s.strip() for s in sentences if isinstance(s, str) and s.strip())
        if not text:
            continue

        label = obj.get("persona", {}).get(attribute, "")
        if not label or label == "Empty":
            continue

        records.append({"text": text, "label": str(label)})

    return records


def load_pastel_age(data_dir: str) -> List[Dict[str, Any]]:
    """Load PASTEL stories labelled by annotator age."""
    return _load_pastel_stories(data_dir, "age")


def load_pastel_gender(data_dir: str) -> List[Dict[str, Any]]:
    """Load PASTEL stories labelled by annotator gender."""
    return _load_pastel_stories(data_dir, "gender")


def load_pastel_ethnic(data_dir: str) -> List[Dict[str, Any]]:
    """Load PASTEL stories labelled by annotator ethnicity."""
    return _load_pastel_stories(data_dir, "ethnic")


def load_pastel_education(data_dir: str) -> List[Dict[str, Any]]:
    """Load PASTEL stories labelled by annotator education level."""
    return _load_pastel_stories(data_dir, "education")


def load_pastel_politics(data_dir: str) -> List[Dict[str, Any]]:
    """Load PASTEL stories labelled by annotator political orientation."""
    return _load_pastel_stories(data_dir, "politics")


def load_pastel_tod(data_dir: str) -> List[Dict[str, Any]]:
    """Load PASTEL stories labelled by annotator-reported time of day."""
    return _load_pastel_stories(data_dir, "tod")
