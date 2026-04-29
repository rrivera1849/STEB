import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List


def _collect_learner_text(
    elem: ET.Element,
    chunks: List[str],
) -> None:
    """
    Depth-first text walker that emits the text written by the learner.

    The FCE error-coding wraps every correction in `<NS><i>...</i><c>...</c></NS>`,
    where `<i>` is the learner's incorrect/original form and `<c>` is the
    examiner's corrected form. For an L1-style signal we want what the
    learner actually wrote, so the entire `<c>` subtree is dropped (its
    `.text` and any descendants) while the `<c>`'s `.tail` is preserved
    because tails belong to the parent's text flow, not the element.

    Args:
        elem: The element whose children to walk.
        chunks: Accumulator list that the collected text fragments are
            appended to in document order.
    """
    for child in elem:
        if child.tag == "c":
            if child.tail:
                chunks.append(child.tail)
            continue
        if child.text:
            chunks.append(child.text)
        _collect_learner_text(child, chunks)
        if child.tail:
            chunks.append(child.tail)


def _extract_learner_text(
    coded_answer: ET.Element,
) -> str:
    """
    Extracts the learner-original text of a single `<coded_answer>` block,
    joining its `<p>` paragraphs with blank lines. Whitespace within each
    paragraph is collapsed.

    Args:
        coded_answer: A `<coded_answer>` element holding `<p>` paragraphs.

    Returns:
        The joined paragraph text, or an empty string if the answer has no
        non-empty paragraphs.
    """
    paragraphs: List[str] = []
    for p in coded_answer.findall("p"):
        chunks: List[str] = []
        if p.text:
            chunks.append(p.text)
        _collect_learner_text(p, chunks)
        cleaned = re.sub(r"\s+", " ", "".join(chunks)).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def load_fce_l1_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the FCE released corpus for L1 (native-language) prediction.

    The First Certificate in English (FCE) released dataset (Yannakoudakis
    et al., 2011) is a Cambridge Learner Corpus subset of upper-intermediate
    ESOL exam scripts. Each script holds two `<coded_answer>` blocks: a
    mandatory Part 1 letter (Q1) and a self-selected Part 2 task (Q2). Both
    are written by the same candidate and labelled with the same L1, read
    from `<head><candidate><personnel><language>`. Each `<coded_answer>` is
    emitted as one record so Q1 and Q2 are independent samples; this
    accepts a mild Q2 topic-by-L1 confound (Q2 prompts are self-selected)
    in exchange for doubling the per-class instance count.

    The corpus is distributed under a non-commercial research/educational
    licence (a `license` file ships in the unpacked tree); see issue #107
    and the cite for Yannakoudakis et al., 2011.

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
        lang_el = learner.find(".//personnel/language")
        if lang_el is None or not (lang_el.text or "").strip():
            continue
        l1 = lang_el.text.strip()
        for coded_answer in learner.iter("coded_answer"):
            text = _extract_learner_text(coded_answer)
            if not text:
                continue
            records.append({"text": text, "label": l1})

    return records
