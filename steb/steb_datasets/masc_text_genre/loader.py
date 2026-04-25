import glob
import os
from typing import Any, Dict, List


SAMPLES_PER_GENRE = 150


def discover_genre_files(
    data_dir: str,
) -> Dict[str, List[str]]:
    """
    Walks the MASC data tree and groups all .txt files by genre (the leaf
    folder name under data/spoken or data/written). Email subdirectories
    (enron/spam/w3c) are recursively flattened into the single 'email' genre.

    Args:
        data_dir: Path to the extracted MASC-3.0.0 root.

    Returns:
        A mapping from genre name to a sorted list of .txt file paths.
    """
    genre_files: Dict[str, List[str]] = {}
    for side in ("spoken", "written"):
        side_root = os.path.join(data_dir, "data", side)
        if not os.path.isdir(side_root):
            continue
        for genre in sorted(os.listdir(side_root)):
            genre_root = os.path.join(side_root, genre)
            if not os.path.isdir(genre_root):
                continue
            files = sorted(glob.glob(os.path.join(genre_root, "**", "*.txt"), recursive=True))
            if files:
                genre_files[genre] = files
    return genre_files


def split_into_equal_chunks(
    text: str,
    n_chunks: int,
) -> List[str]:
    """
    Splits a string into `n_chunks` contiguous character-range chunks of
    as-equal-as-possible length. Original whitespace (including newlines) is
    preserved; chunks may start or end mid-word.

    Args:
        text: Source string for the whole genre.
        n_chunks: Number of chunks to produce.

    Returns:
        A list of `n_chunks` substrings whose concatenation equals the input.
    """
    total = len(text)
    chunks = []
    for i in range(n_chunks):
        start = (i * total) // n_chunks
        end = ((i + 1) * total) // n_chunks
        chunks.append(text[start:end])
    return chunks


def load_masc_text_genre_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the MASC 3.0.0 corpus as a text-genre dataset. Each genre (the leaf
    folder under data/spoken or data/written) contributes exactly 150 samples.
    All .txt files for a genre are concatenated in alphabetic-path order
    (separated by a newline) and the resulting string is split into 150
    equal-length contiguous character ranges.

    Args:
        data_dir: Path to the extracted MASC-3.0.0 root containing data/spoken
            and data/written.

    Returns:
        A list of {'text': str, 'label': str} records, where 'label' is the
        genre name (e.g. 'email', 'newspaper', 'telephone').
    """
    genre_files = discover_genre_files(data_dir)
    records: List[Dict[str, Any]] = []
    for genre in sorted(genre_files):
        texts = []
        for path in genre_files[genre]:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
        joined = "\n".join(texts)
        for chunk in split_into_equal_chunks(joined, SAMPLES_PER_GENRE):
            records.append({"text": chunk, "label": genre})
    return records
