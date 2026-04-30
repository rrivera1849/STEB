import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List


def _learner_text(
    elem: ET.Element,
) -> str:
    """
    Returns the learner-surface text of `elem` and its descendants.

    The FCE error-coding wraps every correction in `<NS><i>...</i><c>...</c></NS>`,
    where `<i>` is the learner's incorrect/original form and `<c>` is the
    examiner's corrected form. For an L1-style signal we want what the
    learner actually wrote, so the entire `<c>` subtree is dropped (its
    `.text` and any descendants) while the `<c>`'s `.tail` is preserved
    because tails belong to the parent's text flow, not the element.

    Args:
        elem: The element to flatten to a string.

    Returns:
        The concatenated learner-surface text in document order, with the
        examiner's `<c>` corrections excluded.
    """
    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if child.tag != "c": # skip corrections
            parts.append(_learner_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _extract_learner_text(
    coded_answer: ET.Element,
) -> str:
    """
    Extracts the learner-original text of a single `<coded_answer>` block,
    joining its `<p>` paragraphs with blank lines.

    Args:
        coded_answer: A `<coded_answer>` element holding `<p>` paragraphs.

    Returns:
        The joined paragraph text.
    """
    paragraphs = [
        re.sub(r"\s+", " ", _learner_text(p)).strip()
        for p in coded_answer.findall("p")
    ]
    return "\n\n".join(p for p in paragraphs if p)


def load_fce_l1_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the FCE released corpus for L1 (native-language) prediction.

    The First Certificate in English (FCE) released dataset (Yannakoudakis
    et al., 2011) at https://aclanthology.org/P11-1019/ is a
    Cambridge Learner Corpus subset of FCE exams written by 16 different
    native language speakers.

    The corpus is distributed under a non-commercial research/educational
    licence.

    Args:
        data_dir: Path to the raw dataset directory containing the
            `fce-released-dataset/` tree as unpacked from the released zip.

    Returns:
        List of records with `text` (one learner answer with `<NS>`
        corrections collapsed to the learner's surface form) and `label`
        (the writer's native language string, e.g. "Spanish") fields.
    """
    root = Path(data_dir)
    dataset_dir = root / "fce-released-dataset" / "dataset"
    if not dataset_dir.is_dir():
        raise ValueError(f"FCE dataset directory not found: {dataset_dir}")

    records: List[Dict[str, Any]] = []
    for xml_path in sorted(dataset_dir.rglob("*.xml")):
        tree = ET.parse(xml_path)
        learner = tree.getroot()
        lang_learner = learner.find(".//personnel/language")
        if lang_learner is None or not (lang_learner.text or "").strip():
            raise ValueError(f"Missing or empty language label in {xml_path}")
        l1 = lang_learner.text.strip()
        for coded_answer in learner.iter("coded_answer"):
            text = _extract_learner_text(coded_answer)
            records.append({"text": text, "label": l1})
    return records
