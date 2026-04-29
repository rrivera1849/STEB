import csv
import os

import pytest

from steb.steb_datasets.ceeces.loader import (
    _extract_letter_text,
    _load_period_by_letter_id,
    load_ceeces_dataset,
)


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_CEECES_DIR = os.path.join(ROOT_DIR, "raw_datasets", "ceeces")
CEECES1_METADATA_PATH = os.path.join(
    RAW_CEECES_DIR, "CEECES1", "CEECES1-metadata.txt"
)

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

# Tests that touch the raw corpus are skipped when it has not been downloaded.
# Run `bash download_datasets.sh` from the repo root to enable them.
requires_raw_dataset = pytest.mark.skipif(
    not os.path.isdir(RAW_CEECES_DIR),
    reason=(
        f"CEECES raw dataset not found at {RAW_CEECES_DIR}; "
        "run download_datasets.sh first."
    ),
)
requires_ceeces1_metadata = pytest.mark.skipif(
    not os.path.isfile(CEECES1_METADATA_PATH),
    reason=(
        f"CEECES1 metadata not found at {CEECES1_METADATA_PATH}; "
        "run download_datasets.sh first."
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ceeces_records():
    """Load CEECES once for this module; skip if the raw corpus is missing."""
    if not os.path.isdir(RAW_CEECES_DIR):
        pytest.skip(
            f"CEECES raw dataset not found at {RAW_CEECES_DIR}; "
            "run download_datasets.sh first."
        )
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

@requires_ceeces1_metadata
def test_period_metadata_parses_known_letter():
    """Metadata loader maps `BOWREY_001` to its known period from CEECES 1."""
    period_by_id = _load_period_by_letter_id(CEECES1_METADATA_PATH)
    assert period_by_id.get("BOWREY_001") == "1680-1699"


@requires_ceeces1_metadata
def test_ceeces1_metadata_header_and_first_rows():
    """
    Snapshot test: the CEECES 1 metadata file as published on Zenodo has 37
    tab-separated columns and a known header. The first three data rows
    correspond to letters BOWREY_001, BOWREY_002, BOWREY_003. This pins the
    file's encoding (CP1252), delimiter (TAB), column ordering, and the
    specific values of fields downstream code may depend on.
    """
    expected_header = [
        "LetterID", "Collection", "Period", "SenderID",
        "SenderFirstName", "SenderLastName", "SenderGender",
        "SenderCurrentRank", "SenderHighestRank", "SenderStatus",
        "SenderAge", "SenderYearOfBirth", "SenderAgeGroup", "SenderRegion",
        "SenderSocialMobility", "SenderEducation", "SenderDNB",
        "RecipientID", "RecipientFirstName", "RecipientLastName",
        "RecipientGender", "RecipientCurrentRank", "RecipientHighestRank",
        "RecipientStatus", "RecipientDNB", "RelationshipCode",
        "Relationship", "LetterAuthenticity", "Year", "YearUncertain",
        "MultipleSenders", "MultipleRecipients", "LetterNotes",
        "CorrespondentNotes", "WordCount", "Source", "PageNumber",
    ]

    with open(CEECES1_METADATA_PATH, "r", encoding="cp1252", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        first_three = [next(reader) for _ in range(3)]

    assert header == expected_header

    bowrey_source = (
        "A Geographical Account of Countries Round the Bay of Bengal, "
        "1669 to 1679, by Thomas Bowrey. Ed. by Sir Richard Carnac Temple. "
        "Cambridge: Hakluyt Society, 1905. Hakluyt Society Second Series "
        "No. XII."
    )
    bowrey_papers_source = (
        "The Papers of Thomas Bowrey 1669-1713. Ed. by Sir Richard Carnac "
        "Temple. London: Hakluyt Society, 1925. Hakluyt Society Second "
        "Series No. LVIII."
    )

    expected_rows = [
        [
            "BOWREY_001", "Bowrey", "1680-1699", "TBOWREY",
            "Thomas", "Bowrey", "M", "M", "M", "merchant",
            "37", "1650", "30-39", "H", "", "",
            "http://www.oxforddnb.com/view/article/57447",
            "JDAVIS", "John", "Davis", "M", "M", "M", "merchant", "",
            "T", "India merchants known to each other", "A",
            "1687", "N", "N", "Y", "",
            "Other recipients: the EIC council in Cuddalore.",
            "696", bowrey_source, "xxxi-xxxiii",
        ],
        [
            "BOWREY_002", "Bowrey", "1680-1699", "TBOWREY",
            "Thomas", "Bowrey", "M", "M", "M", "merchant",
            "37", "1650", "30-39", "H", "", "",
            "http://www.oxforddnb.com/view/article/57447",
            "JDAVIS", "John", "Davis", "M", "M", "M", "merchant", "",
            "T", "India merchants known to each other", "A",
            "1687", "N", "N", "N", "", "",
            "186", bowrey_source, "xxxiii",
        ],
        [
            "BOWREY_003", "Bowrey", "1700-1719", "THAMMOND",
            "Thomas", "Hammond", "M", "M", "M", "merchant",
            "", "", "", "L", "", "", "",
            "TBOWREY", "Thomas", "Bowrey", "M", "M", "M", "merchant",
            "http://www.oxforddnb.com/view/article/57447",
            "T", "business partners", "A",
            "1704", "N", "N", "N", "", "",
            "360", bowrey_papers_source, "149-150",
        ],
    ]

    assert first_three == expected_rows


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
