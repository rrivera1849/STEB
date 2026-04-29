import os

import pytest

from steb.steb_datasets.ceeces.loader import (
    _extract_letter_text,
    _load_period_by_letter_id,
    load_ceeces_dataset,
)


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_CEECES_DIR = os.path.join(ROOT_DIR, "raw_datasets", "ceeces")

# Period values that appear in the CEECES 1 + 2 metadata `Period` column.
EXPECTED_PERIODS = {
    "1640-1659",
    "1660-1679",
    "1680-1699",
    "1700-1719",
    "1720-1739",
    "1740-1759",
    "1760-1779",
    "1780-1800",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ceeces_records():
    """Load CEECES once for this module."""
    assert os.path.exists(
        RAW_CEECES_DIR
    ), "CEECES raw dataset not downloaded; run download_datasets.sh first."
    records = load_ceeces_dataset(RAW_CEECES_DIR)
    assert isinstance(records, list)
    assert records, "CEECES loader should return at least one record"
    return records


# ---------------------------------------------------------------------------
# Tests: dataset-level properties
# ---------------------------------------------------------------------------

def test_record_shape(ceeces_records):
    """Every record has a non-empty string `text` and a string `label`."""
    for rec in ceeces_records:
        assert set(rec.keys()) == {"text", "label"}
        assert isinstance(rec["text"], str) and rec["text"].strip()
        assert isinstance(rec["label"], str) and rec["label"].strip()


def test_labels_are_known_periods(ceeces_records):
    """All emitted labels are values that appear in the CEECES metadata."""
    labels = {rec["label"] for rec in ceeces_records}
    assert labels, "Expected at least one label in CEECES records"
    assert labels.issubset(EXPECTED_PERIODS), (
        f"Unexpected period labels: {labels - EXPECTED_PERIODS}"
    )


def test_total_record_count_is_reasonable(ceeces_records):
    """
    Sanity check: CEECES 1 has ~1180 letters with metadata and CEECES 2 has
    ~1452. We expect roughly that many records, allowing some loss to letters
    without metadata or without paragraph content.
    """
    assert len(ceeces_records) > 2000


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------

def test_period_metadata_parses_known_letter():
    """Metadata loader maps `BOWREY_001` to its known period from CEECES 1."""
    metadata_path = os.path.join(
        RAW_CEECES_DIR, "CEECES1", "CEECES1-metadata.txt"
    )
    period_by_id = _load_period_by_letter_id(metadata_path)
    assert period_by_id.get("BOWREY_001") == "1680-1699"


def test_extract_letter_text_skips_self_closing_notes():
    """`itertext`-based extraction returns paragraph text without editor markup."""
    import xml.etree.ElementTree as ET

    xml = (
        "<TEI xml:id='X_001'>"
        "<text type='letter'>"
        "<p>Porto Novo, <hi>May</hi> 14th, 1687.</p>"
        "<p>To <note resp='editor' value='ANNOTATION'/>Mr. Davis.</p>"
        "<p>   </p>"
        "</text>"
        "</TEI>"
    )
    tei = ET.fromstring(xml)
    text = _extract_letter_text(tei)
    assert text == "Porto Novo, May 14th, 1687.\n\nTo Mr. Davis."
