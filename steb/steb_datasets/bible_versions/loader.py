import os
from typing import Any, Dict, List, Tuple


TRANSLATIONS = ("ASV", "BBE", "DARBY", "DRA", "KJV", "LEB", "WEB", "YLT")


def _parse_chapter_file(
    path: str,
) -> Dict[int, str]:
    """
    Parse a chapter file laid out as one verse per line, where each line is
    prefixed by the verse number, e.g. ``1 In the beginning God created...``.

    Args:
        path: Path to the chapter ``.txt`` file.

    Returns:
        A mapping from verse number to verse text (with the verse-number
        prefix stripped). Lines that do not start with an integer prefix are
        skipped silently.
    """
    verses: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)  # lines look like `1 In the beginning God created...``
            if len(parts) != 2:
                continue
            verse_num_str, verse_text = parts
            try:
                verse_num = int(verse_num_str)
            except ValueError:
                continue
            verse_text = verse_text.strip()
            if verse_text:
                verses[verse_num] = verse_text
    return verses


def _collect_translation(
    translation_dir: str,
) -> Dict[Tuple[str, int, int], str]:
    """
    Walk a single-translation directory and return all verses keyed by the
    canonical ``(book, chapter, verse_num)`` triple.

    The layout is ``<translation>/<Book>/<Book><Chapter>.txt`` (e.g.
    ``KJV/Genesis/Genesis1.txt``).

    Args:
        translation_dir: Absolute path to one translation's root directory.

    Returns:
        A mapping ``(book, chapter, verse_num) -> verse_text``.
    """
    verses: Dict[Tuple[str, int, int], str] = {}
    if not os.path.isdir(translation_dir):
        raise FileNotFoundError(f"{translation_dir} does not exist")

    for book in sorted(os.listdir(translation_dir)):
        book_dir = os.path.join(translation_dir, book)
        if not os.path.isdir(book_dir):
            continue  # elements like .DS_Store
        for filename in sorted(os.listdir(book_dir)):
            if not filename.endswith(".txt"):
                continue
            stem = filename[: -len(".txt")]
            chapter_str = stem[len(book):]
            chapter = int(chapter_str)
            chapter_path = os.path.join(book_dir, filename)
            for verse_num, verse_text in _parse_chapter_file(chapter_path).items():
                verses[(book, chapter, verse_num)] = verse_text

    return verses


def load_bible_versions_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load 8 verse-aligned public-domain English Bible translations from the
    StyleTransferBibleData corpus (Carlson, Riddell & Rockmore, 2018,
    *Royal Society Open Science* 5: 171920).

    The translations span an archaic-to-modern English spectrum, but cover
    the same biblical content. Only references that appear in all 8
    translations are returned.

    The loader returns the full corpus (~31k verses x 8 translations).

    Args:
        data_dir: Path to the ``Data/Bibles`` directory containing one folder
            per translation, e.g. ``raw_datasets/bible_versions/Data/Bibles``.

    Returns:
        A list of ``{'text': verse_text, 'label': translation_name}`` records,
        emitted in deterministic order.
    """
    per_translation: Dict[str, Dict[Tuple[str, int, int], str]] = {} # [str, int, int] = book, chapter, verse
    for translation in TRANSLATIONS:
        translation_dir = os.path.join(data_dir, translation)
        per_translation[translation] = _collect_translation(translation_dir)

    # Intersect references across all translations so every label covers
    # identical biblical content.
    common_refs = set(per_translation[TRANSLATIONS[0]].keys())
    for translation in TRANSLATIONS[1:]:
        common_refs &= set(per_translation[translation].keys())

    # Sort deterministically so loader output order is platform-independent.
    sorted_refs = sorted(common_refs)

    records: List[Dict[str, Any]] = []
    for ref in sorted_refs:
        for translation in TRANSLATIONS:
            records.append({
                "text": per_translation[translation][ref],
                "label": translation,
            })

    return records
