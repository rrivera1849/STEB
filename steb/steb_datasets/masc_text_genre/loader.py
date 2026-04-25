import glob
import os
import re
import warnings
from typing import Any, Dict, List

import numpy as np


SAMPLES_PER_GENRE = 150
MIN_TARGET_WORDS = 40
BLANK_LINE_SPLIT = re.compile(r"\n\s*\n+")


def split_into_blocks(
    text: str,
) -> List[str]:
    """
    Splits a raw .txt file's contents into paragraph-like blocks separated by
    blank lines.

    Lines inside a block are stripped of leading/trailing whitespace; empty
    lines are dropped. If splitting on blank lines yields very few blocks but
    the file has many newlines (e.g. one-tweet-per-line files), the text is
    re-split on single newlines so each non-empty line becomes its own block.

    Args:
        text: Raw file contents.

    Returns:
        A list of non-empty, whitespace-cleaned blocks in document order.
    """
    raw_blocks = BLANK_LINE_SPLIT.split(text)
    blocks = []
    for block in raw_blocks:
        cleaned_lines = [line.strip() for line in block.split("\n") if line.strip()]
        if cleaned_lines:
            blocks.append("\n".join(cleaned_lines))

    if len(blocks) <= 2 and text.count("\n") >= 10:
        line_blocks = [line.strip() for line in text.split("\n") if line.strip()]
        if len(line_blocks) > len(blocks):
            blocks = line_blocks

    return blocks


def split_oversized_block(
    block: str,
    target_words: int,
) -> List[str]:
    """
    Splits a single block that is larger than `target_words` into roughly
    equal word-window pieces of ~target_words words each, preserving word
    order. Returns the original block as a single-element list if it already
    fits within the target.

    Args:
        block: A paragraph-like text block.
        target_words: Per-chunk word target for the genre.

    Returns:
        One or more sub-blocks whose concatenated word sequence equals the
        input.
    """
    words = block.split()
    if len(words) <= target_words:
        return [block]
    n_pieces = max(2, -(-len(words) // target_words))
    edges = np.linspace(0, len(words), n_pieces + 1).round().astype(int).tolist()
    return [" ".join(words[edges[i]:edges[i + 1]]) for i in range(n_pieces) if edges[i + 1] > edges[i]]


def chunk_file_blocks(
    blocks: List[str],
    target_words: int,
) -> List[str]:
    """
    Groups consecutive blocks into chunks whose word counts stay close to
    `target_words` *without exceeding it*. Oversized single blocks are first
    split into target-sized pieces so a single huge paragraph cannot starve
    the rest of the file. The accumulator emits a chunk before adding a block
    that would overshoot the target, so each chunk has ≤ ~target_words words
    and the per-genre candidate pool is ≥ ceil(total_words / target_words).
    A short trailing chunk (< half target) is merged into the previous one.

    Args:
        blocks: Paragraph-like blocks from a single file, in document order.
        target_words: Approximate word count per chunk.

    Returns:
        A list of chunk strings spanning the file from start to end. A
        non-empty input always yields at least one chunk.
    """
    if not blocks:
        return []

    refined: List[str] = []
    for block in blocks:
        refined.extend(split_oversized_block(block, target_words))

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for block in refined:
        block_words = len(block.split())
        if current and current_words + block_words > target_words:
            chunks.append("\n\n".join(current))
            current = [block]
            current_words = block_words
        else:
            current.append(block)
            current_words += block_words

    if current:
        if chunks and current_words < target_words / 2:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current)
        else:
            chunks.append("\n\n".join(current))

    return chunks


def allocate_quotas(
    candidate_counts: List[int],
    total_picks: int,
) -> List[int]:
    """
    Distributes `total_picks` selections across files proportionally to each
    file's candidate-chunk count, guaranteeing at least one pick per file with
    candidates and never exceeding a file's available count.

    Args:
        candidate_counts: Number of candidate chunks per file.
        total_picks: Total picks to distribute (must satisfy
            sum(candidate_counts) >= total_picks).

    Returns:
        A list of per-file quotas summing to exactly `total_picks`.
    """
    total_candidates = sum(candidate_counts)
    quotas = [
        max(1, round(total_picks * c / total_candidates)) if c > 0 else 0
        for c in candidate_counts
    ]
    quotas = [min(q, c) for q, c in zip(quotas, candidate_counts)]

    while sum(quotas) > total_picks:
        idx = max(
            (i for i in range(len(quotas)) if quotas[i] > 1),
            key=lambda i: quotas[i],
            default=None,
        )
        if idx is None:
            break
        quotas[idx] -= 1

    while sum(quotas) < total_picks:
        slack = [c - q for c, q in zip(candidate_counts, quotas)]
        if not any(s > 0 for s in slack):
            break
        idx = max(range(len(slack)), key=lambda i: slack[i])
        quotas[idx] += 1

    return quotas


def pick_evenly_spaced(
    chunks: List[str],
    quota: int,
) -> List[str]:
    """
    Selects `quota` chunks at evenly-spaced indices over the chunk list so the
    selection covers the start, middle, and end of the source file.

    Args:
        chunks: Ordered chunk list for a single file.
        quota: Number of chunks to keep.

    Returns:
        The selected chunks in document order.
    """
    if quota >= len(chunks):
        return list(chunks)
    if quota <= 0:
        return []
    indices = np.linspace(0, len(chunks) - 1, quota).round().astype(int).tolist()
    seen = []
    for idx in indices:
        if not seen or idx != seen[-1]:
            seen.append(idx)
    return [chunks[i] for i in seen]


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


def build_genre_records(
    genre: str,
    files: List[str],
) -> List[Dict[str, Any]]:
    """
    Builds up to SAMPLES_PER_GENRE records for one genre, distributing chunks
    across files and across positions within each file.

    The pipeline is: pre-trim files if there are more than SAMPLES_PER_GENRE,
    derive a per-genre target chunk size, generate all candidate chunks per
    file, allocate per-file quotas, and pick evenly-spaced chunks per file.

    Args:
        genre: Genre label (e.g. 'email', 'telephone').
        files: All .txt file paths discovered for the genre.

    Returns:
        A list of {'text': str, 'label': genre} records.
    """
    if len(files) > SAMPLES_PER_GENRE:
        idx = np.linspace(0, len(files) - 1, SAMPLES_PER_GENRE).round().astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]

    file_blocks: List[List[str]] = []
    file_word_counts: List[int] = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        blocks = split_into_blocks(text)
        file_blocks.append(blocks)
        file_word_counts.append(sum(len(b.split()) for b in blocks))

    total_words = sum(file_word_counts)
    if total_words == 0:
        return []
    target_size = max(MIN_TARGET_WORDS, total_words // SAMPLES_PER_GENRE)

    file_chunks = [chunk_file_blocks(blocks, target_size) for blocks in file_blocks]
    candidate_counts = [len(c) for c in file_chunks]
    total_candidates = sum(candidate_counts)

    if total_candidates <= SAMPLES_PER_GENRE:
        if total_candidates < SAMPLES_PER_GENRE:
            warnings.warn(
                f"Genre '{genre}': only {total_candidates} candidate chunks available "
                f"(< {SAMPLES_PER_GENRE}); emitting all of them.",
                stacklevel=2,
            )
        picks = file_chunks
    else:
        quotas = allocate_quotas(candidate_counts, SAMPLES_PER_GENRE)
        picks = [pick_evenly_spaced(chunks, q) for chunks, q in zip(file_chunks, quotas)]

    records: List[Dict[str, Any]] = []
    for selection in picks:
        for chunk in selection:
            records.append({"text": chunk, "label": genre})
    return records


def load_masc_text_genre_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the MASC 3.0.0 corpus as a text-genre dataset. Each genre (the leaf
    folder under data/spoken or data/written) contributes up to 150 chunks.
    Chunks are produced by splitting each .txt file into paragraph-like blocks
    on blank lines and grouping consecutive blocks until a per-genre target
    word count is reached. Selections are spread across all files in a genre
    and across positions (start/middle/end) within each file.

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
        records.extend(build_genre_records(genre, genre_files[genre]))
    return records
