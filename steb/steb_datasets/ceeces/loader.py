import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List


XML_NS_ID = "{http://www.w3.org/XML/1998/namespace}id"

SUBSETS = (
    ("CEECES1", "CEECES1-metadata.txt", "CEECES 1 - XML files by collection"),
    ("CEECES2", "CEECES2-metadata.txt", "CEECES 2 - XML files by collection"),
)


def _load_period_by_letter_id(
    metadata_path: os.PathLike,
) -> Dict[str, str]:
    """
    Reads the tab-delimited CEECES metadata file and returns a mapping from
    LetterID to Period (e.g. "1680-1699"). Rows with an empty Period are
    skipped (the metadata files have a few trailing blank rows). The metadata
    files Zenodo ships are CP1252-encoded with CRLF line terminators.

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


def _extract_letter_text(
    tei_element: ET.Element,
) -> str:
    """
    Extracts the running text of a single TEI letter element, joining its
    paragraphs with blank lines. Inline editorial markup (notes, page breaks,
    foreign-language spans, highlighting) is flattened to its visible text via
    ElementTree's `itertext`; whitespace within each paragraph is collapsed.

    Args:
        tei_element: A `<TEI>` element representing one letter.

    Returns:
        The joined paragraph text, or an empty string if the letter has no
        non-empty paragraphs.
    """
    text_el = tei_element.find("text")
    if text_el is None:
        return ""
    paragraphs: List[str] = []
    for p in text_el.findall("p"):
        joined = "".join(p.itertext())
        cleaned = re.sub(r"\s+", " ", joined).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def load_ceeces_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the CEECES (Corpus of Early English Correspondence Extension Sampler)
    dataset for historical period prediction. Combines CEECES 1 and CEECES 2
    (the public 18th-century releases of the CEEC-400 project from the
    University of Helsinki) into one dataset, with one record per letter.

    Each TEI XML file under each subset's "XML files by collection" directory
    holds many `<TEI xml:id="LETTER_ID">` letters; the period label for a
    given letter is read from the matching row in the subset's tab-delimited
    metadata file (`Period` column). Letters without a metadata period are
    dropped; periods themselves are passed through verbatim so STEB's
    downstream filtering decides which periods have enough records to keep.

    Args:
        data_dir: Path to the raw dataset directory containing `CEECES1/` and
            `CEECES2/` subdirectories.

    Returns:
        List of records with `text` (one full letter) and `label` (the period
        string from the metadata, e.g. "1680-1699") fields.
    """
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
            for tei in tree.getroot().iter("TEI"):
                letter_id = tei.attrib.get(XML_NS_ID)
                if not letter_id:
                    continue
                period = period_by_id.get(letter_id)
                if not period:
                    continue
                text = _extract_letter_text(tei)
                if not text:
                    continue
                records.append({"text": text, "label": period})

    return records
