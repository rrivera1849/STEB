import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


NLU_TASKS = [
    "logic_bench_yn",
    "logic_bench_mcq",
    "svamp",
    "mbpp",
    "gsm8k",
    "folio",
    "boolq",
    "copa",
    "multirc",
    "sst-2",
    "wsc",
]
DIALECTS = ["aave", "chce", "collsge", "inde", "jame"]    # alphabetical
ALL_LABELS = ["sae"] + DIALECTS                            # SAE original + 5 dialects
HF_SPLIT_NAMES = {                                         # HF uses original-case split names
    "aave": "AAVE",
    "chce": "ChcE",
    "collsge": "CollSgE",
    "inde": "IndE",
    "jame": "JamE",
}
N_PER_TASK = 150            # max source-ids kept per NLU task; loader takes min(N, available)
HF_OWNER = "abhaygupta1266"


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


def _load_one_nlu_task(
    nlu_task: str,
) -> List[Dict[str, Any]]:
    """
    Fetch one EnDive NLU task from Hugging Face and emit per-record entries.

    Loads the 5 dialect splits from ``abhaygupta1266/<nlu_task>``, intersects
    rows by SAE source text (so only sources translated into all 5 dialects
    are kept), deterministically subsamples to at most ``N_PER_TASK``
    source-ids, and emits 6 records per kept source-id: the SAE original
    plus each of the 5 dialect translations.

    Args:
        nlu_task: HF dataset suffix (e.g. ``"svamp"``, ``"sst-2"``).

    Returns:
        Up to ``N_PER_TASK * 6`` records, or fewer if the task has fewer
        than ``N_PER_TASK`` fully-aligned source-ids. Each record is
        ``{"text": str, "label": str}`` where label is one of
        ``{"sae", "aave", "chce", "collsge", "inde", "jame"}``.
    """
    hf_path = f"{HF_OWNER}/{nlu_task}"
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
    if len(fully_aligned) > N_PER_TASK:
        fully_aligned = sorted(rng.sample(fully_aligned, N_PER_TASK))

    records: List[Dict[str, Any]] = []
    for src_text in fully_aligned:
        per_dialect = by_source[src_text]
        records.append({"text": src_text, "label": "sae"})
        for dialect in DIALECTS:
            records.append({"text": per_dialect[dialect], "label": dialect})
    return records


def load_endive(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load the merged EnDive dataset (11 NLU tasks, humaneval excluded).

    Concatenates per-task records from all NLU tasks. Roughly 8,328 records
    total across 6 labels (~1,388 per label), comfortably exceeding STEB's
    benchmark-preset requirement of 150 samples per label at episode_size=3.

    Args:
        data_dir: Unused (HF datasets self-cache); kept for the STEB loader
            calling convention.

    Returns:
        Flat list of records, each ``{"text": str, "label": str}``.
    """
    records: List[Dict[str, Any]] = []
    for nlu_task in NLU_TASKS:
        records.extend(_load_one_nlu_task(nlu_task))
    return records
