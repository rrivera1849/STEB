import os
import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


DIALECTS = ["aave", "chce", "collsge", "inde", "jame"]  # alphabetical, fixed order
HF_SPLIT_NAMES = {                                       # HF uses original-case split names
    "aave": "AAVE",
    "chce": "ChcE",
    "collsge": "CollSgE",
    "inde": "IndE",
    "jame": "JamE",
}
# Keep at least 150 aligned samples per dialect so benchmark settings that
# require episode_size=3 and n_episodes_per_class=50 do not drop all labels.
N_PER_DATASET = 150


def _detect_columns(
    example: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Auto-detect the (source_col, dialect_col) pair for an EnDive row.

    The dialect column always has the form ``"Dialect (<X>)"`` where ``<X>``
    matches the source-column name case-insensitively (e.g. ``"Context"`` /
    ``"Dialect (context)"``, ``"Original"`` / ``"Dialect (Original)"``).

    Args:
        example: One row from any EnDive dialect split.

    Returns:
        ``(source_col, dialect_col)`` — the SAE-original column and the
        dialect-translation column for this NLU task.
    """
    dialect_col = next(c for c in example.keys() if c.startswith("Dialect ("))
    inner = dialect_col[len("Dialect ("):-1]
    source_col = next(c for c in example.keys() if c.lower() == inner.lower())
    return source_col, dialect_col


def load_endive(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load one EnDive NLU-task dataset from Hugging Face.

    Matches parallel rows across the 5 dialect splits via the SAE-source
    column, takes 100 source-ids that exist in all 5 dialects, and emits 500
    records (one per (source_id, dialect)) shaped for STEB's clustering and
    all_to_all_pair_classification tasks.

    The NLU task name is the basename of ``data_dir`` (e.g. ``data_dir`` ends
    in ``"endive/svamp"`` → HF path ``"abhaygupta1266/svamp"``). The directory
    itself does not need to exist on disk; HF's ``datasets`` library handles
    its own caching.

    Args:
        data_dir: Path whose basename names the EnDive NLU task.

    Returns:
        Up to 500 records (or fewer if the task has < 100 source-ids fully
        aligned across all 5 dialects). Each record is
        ``{"text": <dialect translation>, "label": <dialect>}``.
    """
    nlu_task = os.path.basename(data_dir.rstrip("/"))
    hf_path = f"abhaygupta1266/{nlu_task}"

    splits = {
        d: load_dataset(hf_path, split=HF_SPLIT_NAMES[d])
        for d in DIALECTS
    }

    sample_row = splits["aave"][0]
    source_col, dialect_col = _detect_columns(sample_row)

    by_source: Dict[str, Dict[str, str]] = {}
    for dialect, rows in splits.items():
        for row in rows:
            src = row.get(source_col)
            tgt = row.get(dialect_col)
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            by_source.setdefault(src, {})[dialect] = tgt

    fully_aligned = sorted(
        s for s, d in by_source.items() if all(name in d for name in DIALECTS)
    )

    rng = random.Random(42)
    if len(fully_aligned) > N_PER_DATASET:
        fully_aligned = sorted(rng.sample(fully_aligned, N_PER_DATASET))

    records: List[Dict[str, Any]] = []
    for src_text in fully_aligned:
        per_dialect = by_source[src_text]
        for dialect in DIALECTS:
            records.append({
                "text": per_dialect[dialect],
                "label": dialect,
            })
    return records
