from pathlib import Path
from typing import Any, Dict, List


def load_groenwold_aave_sae(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load the Groenwold et al. 2020 AAVE/SAE parallel tweet corpus.

    Reads `aave_samples.txt` and `sae_samples.txt` from `data_dir`. Each line
    in one file is the parallel counterpart of the same-numbered line in the
    other. Returns one record per aligned line-pair, with text ordered
    [aave, sae] (position 0 = most-AAVE, position 1 = least-AAVE) so that
    `order_alignment` interprets the pair as the AAVE-style axis.

    Args:
        data_dir: Path to the dataset directory containing the two .txt files.

    Returns:
        List of records with `text` (ordered [aave_line, sae_line]) and
        `label` ("aave") fields.
    """
    root = Path(data_dir)
    aave_path = root / "aave_samples.txt"
    sae_path = root / "sae_samples.txt"

    for p in (aave_path, sae_path):
        if not p.exists():
            raise ValueError(f"File not found: {p}")

    with aave_path.open("r", encoding="utf-8") as f:
        aave_lines = [line.strip() for line in f]
    with sae_path.open("r", encoding="utf-8") as f:
        sae_lines = [line.strip() for line in f]

    if len(aave_lines) != len(sae_lines):
        raise ValueError(
            f"Line count mismatch: aave={len(aave_lines)}, sae={len(sae_lines)}"
        )

    records: List[Dict[str, Any]] = []
    for aave, sae in zip(aave_lines, sae_lines):
        if not aave or not sae:
            continue
        records.append({"text": [aave, sae], "label": "aave"})

    return records
