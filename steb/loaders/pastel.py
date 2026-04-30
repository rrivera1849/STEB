import glob
import json
import os
from typing import Any, Dict, List


VALID_ATTRIBUTES = ("age", "gender", "country", "ethnic", "education", "politics", "tod")


def load_pastel(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load PASTEL stories labelled by one persona attribute.

    The trailing path component of data_dir names the persona attribute
    to use as the label (e.g. "PASTEL/data/v2/stories/age"). The actual
    JSON files are gathered recursively from the parent directory, so a
    parent like ".../stories" pulls all train/valid/test files together.

    Each story JSON is a single object with keys 'output.sentences' (list
    of 5 sentences rewritten in the annotator's persona) and 'persona'
    (dict with lowercase persona keys). The output sentences are joined
    with single spaces into one text string. Records whose persona
    attribute is missing, an empty string, or the upstream "Empty"
    sentinel are dropped.

    Args:
        data_dir: Path of the form ".../<stories_root>/<attribute>",
            where <attribute> is one of "age", "gender", "country",
            "ethnic", "education", "politics", "tod".

    Returns:
        List of {"text": str, "label": str} records, one per usable story.
    """
    attribute = os.path.basename(data_dir.rstrip(os.sep))
    if attribute not in VALID_ATTRIBUTES:
        raise ValueError(
            f"PASTEL data_dir must end with one of {VALID_ATTRIBUTES}, "
            f"got attribute={attribute!r} from {data_dir!r}"
        )

    stories_dir = os.path.dirname(data_dir.rstrip(os.sep))
    if not os.path.isdir(stories_dir):
        raise FileNotFoundError(f"PASTEL stories directory not found: {stories_dir}")

    files = sorted(glob.glob(os.path.join(stories_dir, "**", "*.json"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .json story files found under {stories_dir}")

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
