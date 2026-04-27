import hashlib
import random
from typing import Any, Dict, Optional


def asset_record_handler(
    example: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Custom record handler for the ASSET dataset.

    Builds one length-2 order-alignment record per HF row by pairing one of
    the 10 crowdsourced simplifications with the original Wikipedia
    sentence. The simplification is sampled with a `random.Random` seeded
    from a stable SHA-256 digest of the original sentence, so the same pair
    is produced on every run (Python's built-in `hash` is randomised per
    process and cannot be used here). Positions are ordered
    simple -> complex, so position 0 is the simpler reference and
    position 1 is the original.

    Args:
        example: A row from the `facebook/asset` (config: simplification)
            dataset, with keys `original` (str) and `simplifications`
            (List[str], length 10).

    Returns:
        `{"text": [simplification, original], "label": "simple"}`, or
        `None` if the row is missing required fields.
    """
    original = example.get("original")
    simplifications = example.get("simplifications")

    if not isinstance(original, str) or not original.strip():
        return None
    if not isinstance(simplifications, list) or not simplifications:
        return None

    seed = int(hashlib.sha256(original.encode("utf-8")).hexdigest(), 16)
    idx = seed % len(simplifications)
    sampled = simplifications[idx]

    if not isinstance(sampled, str) or not sampled.strip():
        return None

    return {
        "text": [sampled.strip(), original.strip()],
        "label": "simple",
    }
