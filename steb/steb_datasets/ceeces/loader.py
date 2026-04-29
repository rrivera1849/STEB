import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

# constant defining expected CEECES XML folder and metadata file names
SUBSETS = (
    ("CEECES1", "CEECES1-metadata.txt", "CEECES 1 - XML files by collection"),
    ("CEECES2", "CEECES2-metadata.txt", "CEECES 2 - XML files by collection"),
)

# Seed used to shuffle records within each label. Fixed so the dataset is
# reproducible across machines and runs.
SHUFFLE_SEED = 42


def _load_period_by_letter_id(
    metadata_path: os.PathLike,
) -> Dict[str, str]:
    """
    Reads the tab-delimited CEECES metadata file and returns a mapping from
    LetterID to Period (e.g. "1680-1699"). T

    Args:
        metadata_path: Path to a CEECESN-metadata.txt file.

    Returns:
        Dict mapping LetterID to Period string.
    """
    period_by_id: Dict[str, str] = {}
    with open(metadata_path, "r", encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            letter_id = (row.get("LetterID") or "").strip()
            period = (row.get("Period") or "").strip()
            if not letter_id or not period:
                continue
            period_by_id[letter_id] = period
    return period_by_id


def _extract_letter_paragraphs(
    tei_element: ET.Element,
) -> List[str]:
    """
    Extracts the paragraphs of a single letter element. Inline editorial
    markup (notes, page breaks, foreign-language spans, highlighting) are
    removed, leaving only the visible text; whitespace within each paragraph
    is collapsed to single spaces.

    Args:
        tei_element: A `<TEI>` element representing one letter.

    Returns:
        List of cleaned paragraph strings. Raises if the letter has no
        non-empty paragraphs.
    """
    text_el = tei_element.find("text")
    paragraphs: List[str] = []
    if text_el is not None:
        for p in text_el.findall("p"):
            joined = "".join(p.itertext())
            cleaned = re.sub(r"\s+", " ", joined).strip()
            if cleaned:
                paragraphs.append(cleaned)
    return paragraphs


def load_ceeces_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the CEECES (Corpus of Early English Correspondence Extension Sampler)
    part 1 and part 2.

    Each TEI XML file under each subset's "XML files by collection" directory
    holds `<TEI xml:id="LETTER_ID">` letters; the period label for a
    given letter is read from the matching row in the subset's tab-delimited
    metadata file.

    Args:
        data_dir: Path to the raw dataset directory containing `CEECES1/` and
            `CEECES2/` subdirectories.

    Returns:
        List of records with `text` (one paragraph) and `label` (the period
        string from the metadata, e.g. "1680-1699") fields.
    """
    # Skip the two pre-1680 periods: each has only one letter in the
    # CEECES 1 metadata.
    SKIP_PERIODS = {"1640-1659", "1660-1679"}

    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Data directory not found: {root}")

    records: List[Dict[str, Any]] = []
    for subset_dir_name, metadata_filename, xml_dir_name in SUBSETS:
        subset_dir = root / subset_dir_name
        metadata_path = subset_dir / metadata_filename
        xml_dir = subset_dir / xml_dir_name
        if not metadata_path.is_file():
            raise ValueError(f"Metadata file not found: {metadata_path}")
        if not xml_dir.is_dir():
            raise ValueError(f"XML directory not found: {xml_dir}")

        period_by_id = _load_period_by_letter_id(metadata_path)
        for xml_path in sorted(xml_dir.glob("*.xml")):
            tree = ET.parse(xml_path)
            for tei in tree.getroot().iter("TEI"):  # several letters per file, saved by author name
                letter_id = tei.attrib.get(
                    "{http://www.w3.org/XML/1998/namespace}id"
                )
                if not letter_id:
                    raise ValueError(f"Letter missing xml:id attribute: {xml_path}")
                period = period_by_id.get(letter_id)
                if not period:
                    raise ValueError(f"Letter ID {letter_id} not found in metadata: {metadata_path}")
                if period in SKIP_PERIODS:
                    continue
                for paragraph in _extract_letter_paragraphs(tei):
                    records.append({"text": paragraph, "label": period})

    # Shuffle so paragraphs from one author/letter are not back-to-back
    # (they were appended in author then letter order above). Fixed seed
    # for reproducibility.
    random.Random(SHUFFLE_SEED).shuffle(records)
    return records
