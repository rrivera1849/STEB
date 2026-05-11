from pathlib import Path
from typing import Any, Dict, List


def _normalise(
    text: str,
) -> str:
    """
    Normalise a sentence for the unchanged-pair check.

    Collapses runs of whitespace and lowercases the result so trivial
    casing/spacing differences between an original and modern line are
    treated as "unchanged".

    Args:
        text: Raw sentence to normalise.

    Returns:
        The whitespace-collapsed, lowercased form of `text`.
    """
    return " ".join(text.split()).lower()


def load_parallel_shakespeare_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load the parallel Shakespeare corpus from Xu et al. (COLING 2012).

    `data_dir` is expected to contain pairs of files named
    `<play>_original.snt.aligned` and `<play>_modern.snt.aligned`, where the
    i-th line of one file is the parallel counterpart of the i-th line of the
    other (source: https://github.com/cocoxu/Shakespeare/tree/master/data/align/plays/merged).

    Each retained record holds an ordered text list `[original, modern]`
    (position 0 = most-Shakespearean, position 1 = least-Shakespearean) so the
    `order_alignment` task interprets the pair as the Shakespearean-vs-modern
    style axis. Pairs whose two sides are identical after lowercasing and
    whitespace collapsing (i.e. the modern version made no real change) are
    skipped, as are pairs where either side is empty.

    Args:
        data_dir: Path to the dataset directory containing the .snt.aligned files.

    Returns:
        List of records with `text` (ordered [original, modern]) and `label`
        ("shakespeare") fields.
    """
    root = Path(data_dir)
    if not root.exists():
        raise ValueError(f"Directory not found: {root}")

    original_paths = sorted(root.glob("*_original.snt.aligned"))
    if not original_paths:
        raise ValueError(f"No '*_original.snt.aligned' files found in {root}")

    records: List[Dict[str, Any]] = []
    for original_path in original_paths:
        play = original_path.name[: -len("_original.snt.aligned")]
        modern_path = root / f"{play}_modern.snt.aligned"
        if not modern_path.exists():
            raise ValueError(f"Missing modern counterpart for {original_path.name}: {modern_path}")

        with original_path.open("r", encoding="utf-8") as f:
            original_lines = [line.strip() for line in f]
        with modern_path.open("r", encoding="utf-8") as f:
            modern_lines = [line.strip() for line in f]

        if len(original_lines) != len(modern_lines):
            raise ValueError(
                f"Line count mismatch for {play}: "
                f"original={len(original_lines)}, modern={len(modern_lines)}"
            )

        for original, modern in zip(original_lines, modern_lines):
            if not original or not modern:
                continue
            if _normalise(original) == _normalise(modern):
                continue
            records.append({"text": [original, modern], "label": "shakespeare"})

    return records
