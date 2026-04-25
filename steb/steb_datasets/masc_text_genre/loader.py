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
    words: List[str],
    n_chunks: int,
) -> List[str]:
    """
    Splits a word list into `n_chunks` contiguous chunks of as-equal-as-possible
    length, joined back with single spaces.

    Args:
        words: Whitespace-tokenized word list for the whole genre.
        n_chunks: Number of chunks to produce.

    Returns:
        A list of `n_chunks` strings whose concatenated words equal the input.
    """
    total = len(words)
    chunks = []
    for i in range(n_chunks):
        start = (i * total) // n_chunks
        end = ((i + 1) * total) // n_chunks
        chunks.append(" ".join(words[start:end]))
    return chunks


def load_masc_text_genre_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the MASC 3.0.0 corpus as a text-genre dataset. Each genre (the leaf
    folder under data/spoken or data/written) contributes exactly 150 samples.
    All .txt files for a genre are concatenated in alphabetic-path order, the
    result is whitespace-tokenized, and the tokens are split into 150
    equal-length contiguous chunks.

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
        words = " ".join(texts).split()
        for chunk in split_into_equal_chunks(words, SAMPLES_PER_GENRE):
            records.append({"text": chunk, "label": genre})
    return records
