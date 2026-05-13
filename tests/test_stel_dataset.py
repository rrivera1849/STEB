import csv
import os
from typing import Dict, Set, Tuple

import pytest

from steb.loaders.stel import (
    load_stel,
    extract_pairs_from_row,
    parse_id_component,
    parse_stel_id,
    split_id_components,
)


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_STEL_DIR = os.path.join(ROOT_DIR, "raw_datasets", "STEL")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stel_records():
    """Load STEL once for this module."""
    assert os.path.exists(
        RAW_STEL_DIR
    ), "STEL raw dataset not downloaded; run download_datasets.sh first."
    records = load_stel(RAW_STEL_DIR)
    assert isinstance(records, list)
    assert records, "STEL loader should return at least one record"
    return records


@pytest.fixture(scope="module")
def stel_pairs_by_label(stel_records):
    """Map style label -> set of (most, least) pairs from processed STEL records."""
    by_label: Dict[str, Set[Tuple[str, str]]] = {}
    for rec in stel_records:
        text = rec["text"]
        label = rec["label"]
        assert isinstance(text, list) and len(text) == 2
        by_label.setdefault(label, set()).add((text[0], text[1]))
    return by_label


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _first_row_from_tsv(tsv_path: str) -> Dict[str, str]:
    assert os.path.exists(tsv_path), f"Expected STEL TSV file not found: {tsv_path}"
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return next(reader)  # first data row after header


# ---------------------------------------------------------------------------
# Tests: dataset-level properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,expected_count",
    [
        ("formality", 196),
        ("simplicity", 197),
        ("contraction", 101),
        ("emotives", 200),
        ("nbr_substitution", 100),
    ],
)
def test_stel_label_counts_match_expected(stel_pairs_by_label, label, expected_count):
    """Sanity check: known number of unique pairs per style label."""
    assert len(stel_pairs_by_label.get(label, set())) == expected_count


# ---------------------------------------------------------------------------
# Tests: row-level extraction for specific TSVs
# ---------------------------------------------------------------------------

class TestSTELFirstRowExtraction:
    """Row-level tests: specific first-line examples from each STEL TSV."""

    @pytest.fixture(scope="class")
    def stel_root(self):
        stel_root = os.path.join(RAW_STEL_DIR, "Data", "STEL")
        if not os.path.exists(stel_root):
            stel_root = RAW_STEL_DIR
        return stel_root

    def _check_example(
        self,
        stel_root: str,
        relative_path: str,
        style_type: str,
        expected_most_id_component: str,
        expected_least_id_component: str,
        expected_most: str,
        expected_least: str,
        by_label: Dict[str, Set[Tuple[str, str]]],
    ):
        tsv_path = os.path.join(stel_root, relative_path)
        row = _first_row_from_tsv(tsv_path)

        # 1) Direct extraction from raw row
        pairs = extract_pairs_from_row(row, style_type)
        assert pairs, f"No pairs extracted from first row of {os.path.basename(tsv_path)}"
        assert (expected_most, expected_least) in pairs

        # 2) Presence in processed dataset
        label_pairs = by_label.get(style_type, set())
        assert (expected_most, expected_least) in label_pairs

        # 3) ID → position mapping: expected components have correct style ordering
        id_str = row.get("ID", "").strip()
        assert id_str, f"Missing ID for first row in {os.path.basename(tsv_path)}"

        parsed = parse_stel_id(id_str)
        assert parsed is not None, f"Unexpected ID format in {os.path.basename(tsv_path)}: {id_str}"
        anchor_ids, _ = parsed

        components = split_id_components(anchor_ids, style_type)
        assert len(components) == 2, f"Expected two components in anchor IDs, got {components}"
        assert expected_most_id_component in components
        assert expected_least_id_component in components

        most_parsed = parse_id_component(expected_most_id_component, style_type)
        least_parsed = parse_id_component(expected_least_id_component, style_type)
        assert most_parsed is not None and least_parsed is not None

        _, most_pos = most_parsed
        _, least_pos = least_parsed
        assert most_pos == 0, f"{expected_most_id_component} should be position 0 (most style)"
        assert least_pos == 1, f"{expected_least_id_component} should be position 1 (least style)"

    def test_formality_first_row(self, stel_root, stel_pairs_by_label):
        self._check_example(
            stel_root,
            os.path.join("dimensions", "quad_stel-dimension_formal-100_sample.tsv"),
            "formality",
            "f-454",
            "i-454",
            "He has a very distinct walk.",
            "But has a lil slang 2 his walk.",
            stel_pairs_by_label,
        )

    def test_simplicity_first_row(self, stel_root, stel_pairs_by_label):
        self._check_example(
            stel_root,
            os.path.join("dimensions", "quad_stel-dimension_simple-100_sample.tsv"),
            "simplicity",
            "s-t3-240",
            "c-240",
            "Human skin can change from very dark brown to very pale pink.",
            "Human skin hues can range from very dark brown to very pale pink.",
            stel_pairs_by_label,
        )

    def test_contraction_first_row(self, stel_root, stel_pairs_by_label):
        self._check_example(
            stel_root,
            os.path.join("characteristics", "quad_questions_char_contraction.tsv"),
            "contraction",
            "ction-0",
            "wiki-0",
            "The line's name derives from its use in the Medieval French Roman d'Alexandre of 1170, although it'd already been used several decades earlier in Le Pèlerinage de Charlemagne.",
            "The line's name derives from its use in the Medieval French Roman d'Alexandre of 1170, although it had already been used several decades earlier in Le Pèlerinage de Charlemagne.",
            stel_pairs_by_label,
        )

    def test_emotives_first_row(self, stel_root, stel_pairs_by_label):
        self._check_example(
            stel_root,
            os.path.join("characteristics", "quad_questions_char_emotives.tsv"),
            "emotives",
            "emote0",
            "emoji0",
            "proud :) LINK/MEDIA",
            "proud 😀 LINK/MEDIA",
            stel_pairs_by_label,
        )

    def test_substitution_first_row(self, stel_root, stel_pairs_by_label):
        self._check_example(
            stel_root,
            os.path.join("characteristics", "quad_questions_char_substitution.tsv"),
            "nbr_substitution",
            "leet-0",
            "norm-0",
            "<3 friends 4ever",
            "<3 friends forever",
            stel_pairs_by_label,
        )

