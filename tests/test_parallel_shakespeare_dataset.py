import os
from typing import Dict, List

import pytest

from steb.steb_datasets.parallel_shakespeare.loader import (
    _normalise,
    load_parallel_shakespeare_dataset,
)


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT_DIR, "raw_datasets", "parallel_shakespeare")


@pytest.fixture(scope="module")
def shakespeare_records():
    """Load the parallel Shakespeare dataset once for this module."""
    if not os.path.exists(RAW_DIR):
        pytest.skip(
            f"Raw parallel_shakespeare dataset not found at {RAW_DIR}; "
            "run download_datasets.sh first."
        )
    records = load_parallel_shakespeare_dataset(RAW_DIR)
    assert isinstance(records, list)
    assert records, "Loader should return at least one record"
    return records


def test_total_record_count(shakespeare_records):
    """Locks in the total number of kept (non-identical) parallel pairs across all 17 plays."""
    assert len(shakespeare_records) == 19746


def test_record_schema(shakespeare_records):
    """Every record is a [original, modern] pair labelled 'shakespeare'."""
    for rec in shakespeare_records:
        text = rec["text"]
        assert isinstance(text, list) and len(text) == 2
        assert isinstance(text[0], str) and isinstance(text[1], str)
        assert text[0] and text[1]
        assert rec["label"] == "shakespeare"


def test_unchanged_pairs_are_filtered(shakespeare_records):
    """The case-insensitive, whitespace-collapsed unchanged-pair filter actually fired."""
    for rec in shakespeare_records:
        original, modern = rec["text"]
        assert _normalise(original) != _normalise(modern), (
            f"Unchanged pair leaked through filter: {original!r} vs {modern!r}"
        )


@pytest.fixture(scope="module")
def records_by_play() -> Dict[str, List[List[str]]]:
    """
    Group records by play, recovered from the raw aligned files. We re-derive
    counts from the raw data here rather than threading play-id through the
    loader (the loader's contract intentionally only exposes text + label).
    """
    if not os.path.exists(RAW_DIR):
        pytest.skip("Raw dataset missing")
    from pathlib import Path

    grouped: Dict[str, List[List[str]]] = {}
    for original_path in sorted(Path(RAW_DIR).glob("*_original.snt.aligned")):
        play = original_path.name[: -len("_original.snt.aligned")]
        modern_path = Path(RAW_DIR) / f"{play}_modern.snt.aligned"
        original_lines = [line.strip() for line in original_path.open(encoding="utf-8")]
        modern_lines = [line.strip() for line in modern_path.open(encoding="utf-8")]
        kept = []
        for original, modern in zip(original_lines, modern_lines):
            if not original or not modern:
                continue
            if _normalise(original) == _normalise(modern):
                continue
            kept.append([original, modern])
        grouped[play] = kept
    return grouped


@pytest.mark.parametrize(
    "play,expected_count",
    [
        ("hamlet", 1217),
        ("macbeth", 876),
        ("romeojuliet", 1363),
    ],
)
def test_per_play_pair_counts(records_by_play, play, expected_count):
    """Sanity check: known number of kept pairs per play."""
    assert len(records_by_play[play]) == expected_count


def test_hamlet_first_kept_pair(records_by_play, shakespeare_records):
    """Spot-check: the first non-identical Hamlet line is the well-known opening exchange."""
    expected = ["Excellent well.", "Of course."]
    assert records_by_play["hamlet"][0] == expected
    assert {"text": expected, "label": "shakespeare"} in shakespeare_records
