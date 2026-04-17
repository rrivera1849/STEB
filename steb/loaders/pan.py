import glob
import json
import os
import random
from typing import Any, Dict, List

def load_pan15_dataset(data_dir):
    """
    Loads the PAN15 authorship verification dataset.
    
    Expected structure:
    data_dir/
      truth.txt
      EN001/
        known01.txt
        unknown.txt
      EN002/
      ...
    """
    truth_path = os.path.join(data_dir, "truth.txt")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth.txt not found in {data_dir}")

    # Read truth file
    # Format: EN001 Y
    problem_labels = {}
    with open(truth_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pid = parts[0]
                label_char = parts[1]
                # Y -> 1 (Same), N -> 0 (Different)
                problem_labels[pid] = 1 if label_char == 'Y' else 0

    # Iterate over problems
    # To rely on sequential grouping, we must yield pairs strictly together.
    # We sort keys to ensure deterministic order if that matters, but crucial is (A,B) order.
    
    sorted_pids = sorted(problem_labels.keys())
    
    samples = []
    for pid in sorted_pids:
        problem_dir = os.path.join(data_dir, pid)
        if not os.path.exists(problem_dir):
            continue
            
        known_path = os.path.join(problem_dir, "known01.txt")
        unknown_path = os.path.join(problem_dir, "unknown.txt")
        
        if not os.path.exists(known_path) or not os.path.exists(unknown_path):
            continue
            
        try:
            with open(known_path, "r", encoding="utf-8", errors="replace") as f:
                text_a = f.read()
            with open(unknown_path, "r", encoding="utf-8", errors="replace") as f:
                text_b = f.read()
        except Exception:
            # Skip if read error
            continue
            
        if problem_labels[pid] == 1:
            label_str = f"trial_{pid}_true"
        else:
            label_str = f"trial_{pid}_false"
        
        # Yield Text A then Text B
        samples.append({"text": text_a, "label": label_str})
        samples.append({"text": text_b, "label": label_str})

    return samples


def load_pan13_dataset(
    data_dir: str,
) -> list[dict[str, str]]:
    """
    Loads a PAN13 authorship verification dataset for a single language.

    The *data_dir* argument encodes the corpus path and the target
    language, e.g. ``…/pan13-test/EN``.  The last component is used as
    a prefix filter (``EN``, ``GR``, or ``SP``) and the parent
    directory is expected to contain ``truth.txt`` and the problem
    subdirectories.

    PAN13 problems may contain multiple known-author documents
    (``known01.txt`` … ``knownNN.txt``).  All known documents are
    concatenated (separated by double newlines) into a single text
    that forms one side of the verification pair; ``unknown.txt``
    forms the other.

    Args:
        data_dir: Path whose last component is the language prefix
                  (e.g. ``…/pan13-corpus/EN``).

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs (known, unknown) per problem.
    """
    corpus_dir = os.path.dirname(data_dir)
    language_prefix = os.path.basename(data_dir)

    truth_path = os.path.join(corpus_dir, "truth.txt")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth.txt not found in {corpus_dir}")

    problem_labels: dict[str, int] = {}
    with open(truth_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pid = parts[0]
                label_char = parts[1]
                problem_labels[pid] = 1 if label_char == "Y" else 0

    sorted_pids = sorted(
        pid for pid in problem_labels
        if pid.startswith(language_prefix)
    )

    samples: list[dict[str, str]] = []
    for pid in sorted_pids:
        problem_dir = os.path.join(corpus_dir, pid)
        if not os.path.exists(problem_dir):
            continue

        unknown_path = os.path.join(problem_dir, "unknown.txt")
        if not os.path.exists(unknown_path):
            continue

        known_files = sorted(
            f for f in os.listdir(problem_dir)
            if f.startswith("known") and f.endswith(".txt")
        )
        if not known_files:
            continue

        try:
            known_texts = []
            for kf in known_files:
                with open(os.path.join(problem_dir, kf), "r", encoding="utf-8", errors="replace") as f:
                    known_texts.append(f.read())
            text_a = "\n\n".join(known_texts)

            with open(unknown_path, "r", encoding="utf-8", errors="replace") as f:
                text_b = f.read()
        except Exception:
            continue

        if problem_labels[pid] == 1:
            label_str = f"trial_{pid}_true"
        else:
            label_str = f"trial_{pid}_false"

        samples.append({"text": text_a, "label": label_str})
        samples.append({"text": text_b, "label": label_str})

    return samples


def load_pan_jsonl_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a PAN authorship verification dataset stored as JSONL files.

    Works with PAN20, PAN21, and any future edition that uses the same
    format.  The directory must contain exactly two ``.jsonl`` files:
    one whose name ends with ``-truth.jsonl`` (the labels) and one
    that does not (the data).

    Each data record has an ``id``, ``fandoms``, and ``pair`` (a list
    of two texts).  Each truth record has an ``id`` and ``same``
    (boolean).

    The dataset is expected to have been subsampled at download time
    (see ``download_datasets.sh``).

    Args:
        data_dir: Path to the directory containing the JSONL files.

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs (text_a, text_b) per problem.
    """
    jsonl_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".jsonl")
    )
    truth_files = [f for f in jsonl_files if f.endswith("-truth.jsonl")]
    data_files = [f for f in jsonl_files if not f.endswith("-truth.jsonl")]

    if len(truth_files) != 1 or len(data_files) != 1:
        raise FileNotFoundError(
            f"Expected one data and one truth JSONL file in {data_dir}, "
            f"found {jsonl_files}"
        )

    data_path = os.path.join(data_dir, data_files[0])
    truth_path = os.path.join(data_dir, truth_files[0])

    with open(truth_path, "r") as f:
        truth_by_id = {
            t["id"]: t["same"]
            for t in (json.loads(line) for line in f if line.strip())
        }

    samples: List[Dict[str, Any]] = []
    with open(data_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record["id"]
            pair = record["pair"]

            if pid not in truth_by_id or len(pair) != 2:
                continue

            same = truth_by_id[pid]
            label_str = f"trial_{pid}_true" if same else f"trial_{pid}_false"

            samples.append({"text": pair[0], "label": label_str})
            samples.append({"text": pair[1], "label": label_str})

    return samples


def load_pan24_generative_authorship(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the PAN24 Generative Authorship (news) dataset.

    Expects the directory layout produced by unzipping the official release:

        {data_dir}/human.jsonl
        {data_dir}/machines/<model>.jsonl

    Each ``.jsonl`` file contains records of the form ``{"id", "text"}``. Human
    records are assigned the label ``"human"``, and each machine record is
    assigned a label equal to the basename of its file (without the ``.jsonl``
    extension). Empty and whitespace-only texts are skipped.

    Args:
        data_dir: Path to the extracted dataset directory.

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    records: List[Dict[str, Any]] = []

    files = [("human", os.path.join(data_dir, "human.jsonl"))]
    machines_dir = os.path.join(data_dir, "machines")
    for fname in sorted(os.listdir(machines_dir)):
        if not fname.endswith(".jsonl"):
            continue
        label = fname[: -len(".jsonl")]
        files.append((label, os.path.join(machines_dir, fname)))

    for label, path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                records.append({"text": text, "label": label})

    return records


def load_pan25_26_generative_ai_detection(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a genre split of the PAN25/26 Generative AI Detection (Task 1) validation set.

    The same underlying data is used by both the PAN25 and PAN26
    editions of the shared task, which is why the dataset is labelled
    ``PAN25_26``.

    The *data_dir* argument encodes both the dataset root and the target
    genre, e.g. ``{raw_datasets}/pan25-26-generative-ai-detection-task1/news``.
    The last path component (``news``, ``fiction``, or ``essays``) is
    used to filter records; the parent directory must contain
    ``val.jsonl``.

    Each record in ``val.jsonl`` has the fields ``id``, ``text``,
    ``model``, ``label``, and ``genre``. This loader returns one record
    per matching row, labelled by ``model`` (yielding a multi-class
    clustering dataset with human + all LLMs in the selected genre).

    Args:
        data_dir: Path whose last component is the genre name.

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    base_dir = os.path.dirname(data_dir)
    genre = os.path.basename(data_dir)

    val_path = os.path.join(base_dir, "val.jsonl")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"val.jsonl not found in {base_dir}")

    records: List[Dict[str, Any]] = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("genre") != genre:
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            records.append({"text": text, "label": row["model"]})

    return records


def load_pan18_style_change(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the PAN18 Style Change Detection validation set as a
    pre-defined pair classification task.

    PAN18 is document-level: each ``problem-*.truth`` file contains a
    boolean ``changes`` flag (does the document have any style change?)
    and, when true, a list of ``positions`` giving the character offsets
    of the first non-whitespace character of each new segment.

    We map each document onto exactly one authorship-verification trial
    by splitting it into two halves:

    * ``changes == false`` -> split at the whitespace boundary nearest
      the character midpoint of the document. Both halves are by the
      same author, yielding a positive pair (label suffix ``true``).
    * ``changes == true``  -> take the first segment boundary. Side A
      is ``text[:positions[0]]``; side B is
      ``text[positions[0]:positions[1]]`` if a second position exists,
      else ``text[positions[0]:]``. These two adjacent segments are by
      different authors, yielding a negative pair (label suffix
      ``false``). For multi-change documents the second segment is
      truncated to the next boundary so both sides correspond to a
      single (distinct) author.

    The validation set ships perfectly balanced (746/746), so no
    downsampling is needed.

    Args:
        data_dir: Path to the directory containing ``problem-*.txt``
                  and ``problem-*.truth`` files.

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs (text_a, text_b) per document. Each trial has a unique
        label of the form ``trial_{doc_id}_{true|false}``.
    """
    problem_files = sorted(glob.glob(os.path.join(data_dir, "problem-*.txt")))
    if not problem_files:
        raise FileNotFoundError(f"No problem-*.txt files found in {data_dir}")

    samples: List[Dict[str, Any]] = []
    for problem_path in problem_files:
        doc_id = os.path.basename(problem_path)[len("problem-"):-len(".txt")]
        truth_path = os.path.join(data_dir, f"problem-{doc_id}.truth")
        if not os.path.exists(truth_path):
            raise FileNotFoundError(f"Truth file not found for {problem_path}")

        with open(problem_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        with open(truth_path, "r") as f:
            truth = json.load(f)

        if truth.get("changes"):
            positions = truth.get("positions", [])
            if not positions:
                raise ValueError("Positions not found for truth file...", truth_path)
            split = positions[0]
            end = positions[1] if len(positions) > 1 else len(text)
            text_a = text[:split]
            text_b = text[split:end]
            label_str = f"trial_{doc_id}_false"
        else:
            mid = len(text) // 2
            # Snap to nearest whitespace boundary so both halves start on
            # a word boundary (matching how PAN18 positions are defined).
            split = mid
            for offset in range(len(text)):
                left = mid - offset
                right = mid + offset
                if left >= 0 and text[left].isspace():
                    split = left + 1
                    break
                if right < len(text) and text[right].isspace():
                    split = right + 1
                    break
            text_a = text[:split]
            text_b = text[split:]
            label_str = f"trial_{doc_id}_true"

        if not text_a.strip() or not text_b.strip():
            continue

        samples.append({"text": text_a, "label": label_str})
        samples.append({"text": text_b, "label": label_str})

    return samples


def _load_pan_style_change_dataset(
    data_dir: str,
    balance: bool,
) -> List[Dict[str, Any]]:
    """
    Shared implementation for PAN Style Change Detection pair classification.

    Reads the validation split for a single difficulty level (``easy``,
    ``medium``, or ``hard``). Each source document is a sentence-segmented
    text (one sentence per line) accompanied by a
    ``truth-problem-<id>.json`` file whose ``changes`` list flags, for
    every pair of consecutive sentences, whether the author changed
    (``1``) or not (``0``). Each consecutive sentence pair becomes an
    authorship verification trial:

    * ``changes[i] == 0`` (same author)       -> label suffix ``true``
    * ``changes[i] == 1`` (different author)  -> label suffix ``false``

    When ``balance`` is True, a per-document stratified downsampling is
    applied: all different-author pairs are kept, and same-author pairs
    are sampled at the global rate
    ``f = n_diff_total / n_same_total`` so that the per-document
    contribution stays proportional to its length and the overall pair
    counts are balanced. Sampling is seeded with ``random.Random(42)``
    for reproducibility.

    The *data_dir* argument encodes both the dataset root and the
    difficulty level, e.g. ``{raw_datasets}/pan26-style-change/easy``.
    That directory is expected to contain the validation
    ``problem-*.txt`` and ``truth-problem-*.json`` files directly.

    Args:
        data_dir: Path whose last component is the difficulty level.
        balance: Whether to perform balanced per-document downsampling.

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs (sentence_a, sentence_b) per trial. Each trial has a
        unique label of the form
        ``trial_{difficulty}_{doc_id}_{pair_idx}_{true|false}``.
    """
    difficulty = os.path.basename(data_dir)

    problem_files = sorted(glob.glob(os.path.join(data_dir, "problem-*.txt")))
    if not problem_files:
        raise FileNotFoundError(f"No problem-*.txt files found in {data_dir}")

    per_doc: List[Dict[str, Any]] = []
    total_same = 0
    total_diff = 0
    for problem_path in problem_files:
        doc_id = os.path.basename(problem_path)[len("problem-"):-len(".txt")]
        truth_path = os.path.join(data_dir, f"truth-problem-{doc_id}.json")
        if not os.path.exists(truth_path):
            raise FileNotFoundError(f"Truth file not found for {problem_path}")

        with open(problem_path, "r", encoding="utf-8", errors="replace") as f:
            sentences = [line for line in f.read().splitlines() if line.strip()]
        with open(truth_path, "r") as f:
            truth = json.load(f)
        changes = truth.get("changes", [])

        if len(sentences) < 2 or len(changes) != len(sentences) - 1:
            continue

        same_idxs = [i for i, c in enumerate(changes) if c == 0]
        diff_idxs = [i for i, c in enumerate(changes) if c == 1]
        total_same += len(same_idxs)
        total_diff += len(diff_idxs)
        per_doc.append({
            "doc_id": doc_id,
            "sentences": sentences,
            "same_idxs": same_idxs,
            "diff_idxs": diff_idxs,
        })

    if total_diff == 0 or total_same == 0:
        raise ValueError(
            f"Expected both same-author and different-author pairs in {data_dir}"
        )

    rng = random.Random(42)
    sample_rate = total_diff / total_same if balance else 1.0

    samples: List[Dict[str, Any]] = []
    for doc in per_doc:
        sentences = doc["sentences"]
        doc_id = doc["doc_id"]

        if balance:
            n_same_keep = min(
                len(doc["same_idxs"]),
                round(sample_rate * len(doc["same_idxs"])),
            )
            kept_same = rng.sample(doc["same_idxs"], n_same_keep) if n_same_keep > 0 else []
        else:
            kept_same = doc["same_idxs"]
        kept_diff = doc["diff_idxs"]

        for pair_idx in kept_same:
            label_str = f"trial_{difficulty}_{doc_id}_{pair_idx}_true"
            samples.append({"text": sentences[pair_idx], "label": label_str})
            samples.append({"text": sentences[pair_idx + 1], "label": label_str})
        for pair_idx in kept_diff:
            label_str = f"trial_{difficulty}_{doc_id}_{pair_idx}_false"
            samples.append({"text": sentences[pair_idx], "label": label_str})
            samples.append({"text": sentences[pair_idx + 1], "label": label_str})

    return samples


def load_pan22_style_change(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a task split of the PAN22 Style Change Detection validation set
    as a pre-defined pair classification task.

    PAN22 ships three different tasks (not difficulty levels):

    * ``basic`` (dataset1): paragraph-level, exactly two authors, a
      single style change per document.
    * ``advanced`` (dataset2): paragraph-level, 1–5 authors, multiple
      possible style changes.
    * ``sentence`` (dataset3): sentence-level, 1–5 authors, multiple
      possible style changes.

    All three tasks share the same truth-file shape: a binary
    ``changes`` array over consecutive text units (paragraphs for
    ``basic``/``advanced``, sentences for ``sentence``). Pairs are
    emitted unchanged (no downsampling). See
    :func:`_load_pan_style_change_dataset` for the trial label
    convention.

    Args:
        data_dir: Path whose last component is the task split
                  (``basic``, ``advanced``, or ``sentence``).

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs per trial.
    """
    return _load_pan_style_change_dataset(data_dir, balance=False)


def load_pan23_style_change(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a difficulty split of the PAN23 Style Change Detection
    (Multi-Author Writing Style Analysis) validation set as a
    pre-defined pair classification task.

    PAN23 operates at the **paragraph** level: each line in a
    ``problem-*.txt`` file is a full paragraph, and the corresponding
    ``truth-problem-*.json`` ``changes`` list flags, for every pair of
    consecutive paragraphs, whether the author changed. All consecutive
    paragraph pairs are emitted unchanged (no downsampling). See
    :func:`_load_pan_style_change_dataset` for the trial label
    convention.

    Args:
        data_dir: Path whose last component is the difficulty level
                  (``easy``, ``medium``, or ``hard``).

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs per trial.
    """
    return _load_pan_style_change_dataset(data_dir, balance=False)


def load_pan24_style_change(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a difficulty split of the PAN24 Style Change Detection
    (Multi-Author Writing Style Analysis) validation set as a
    pre-defined pair classification task.

    PAN24 operates at the **paragraph** level (not sentence): each line
    in a ``problem-*.txt`` file is a full paragraph, and the
    corresponding ``truth-problem-*.json`` ``changes`` list flags,
    for every pair of consecutive paragraphs, whether the author
    changed. All consecutive paragraph pairs are emitted unchanged
    (no downsampling). See :func:`_load_pan_style_change_dataset`
    for the trial label convention.

    Args:
        data_dir: Path whose last component is the difficulty level
                  (``easy``, ``medium``, or ``hard``).

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs per trial.
    """
    return _load_pan_style_change_dataset(data_dir, balance=False)


def load_pan25_style_change(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a difficulty split of the PAN25 Style Change Detection
    (Multi-Author Writing Style Analysis) validation set as a
    pre-defined pair classification task.

    All consecutive sentence pairs from every validation document are
    emitted unchanged (no downsampling). See
    :func:`_load_pan_style_change_dataset` for details of the format
    and trial label convention.

    Args:
        data_dir: Path whose last component is the difficulty level
                  (``easy``, ``medium``, or ``hard``).

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs per trial.
    """
    return _load_pan_style_change_dataset(data_dir, balance=False)


def load_pan26_style_change(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads a difficulty split of the PAN26 Style Change Detection
    (Multi-Author Writing Style Analysis) validation set as a
    pre-defined pair classification task.

    The raw PAN26 pair distribution is heavily imbalanced toward the
    same-author class (up to ~96% in the medium split), which would
    inflate embedding cost. This loader applies a per-document stratified
    downsampling that keeps all different-author pairs and samples
    same-author pairs per document so that the overall pair counts are
    balanced. See :func:`_load_pan_style_change_dataset` for details.

    Args:
        data_dir: Path whose last component is the difficulty level
                  (``easy``, ``medium``, or ``hard``).

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs per trial.
    """
    return _load_pan_style_change_dataset(data_dir, balance=True)


def load_pan25_collaborative_text_classification(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Loads the PAN25 Task 2 Human-AI Collaborative Text Classification dev set.

    Expects ``dev.jsonl`` inside *data_dir*. Each record contains
    ``text`` and ``label_text``, where ``label_text`` identifies one of
    six human-AI collaboration categories (e.g. ``"fully human-written"``,
    ``"human-initiated, then machine-continued"``). Records are labelled
    by ``label_text`` to produce a six-class clustering dataset. Empty
    and whitespace-only texts are skipped.

    Args:
        data_dir: Path to the directory containing ``dev.jsonl``.

    Returns:
        A list of records with ``text`` and ``label`` keys.
    """
    dev_path = os.path.join(data_dir, "dev.jsonl")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"dev.jsonl not found in {data_dir}")

    records: List[Dict[str, Any]] = []
    with open(dev_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            records.append({"text": text, "label": row["label_text"]})

    return records


def load_enron_authorship_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """Load the Enron Authorship Corpus for authorship verification.

    Each problem is a subdirectory named ``Author.name[Y]`` or
    ``Author.name[N]``.  ``[Y]`` means the unknown text was written by
    the same author as the known texts; ``[N]`` means it was not.

    Known-author files are named ``known - Author - Mail_N.txt`` and the
    unknown file is named ``unknown - ... - Mail_N.txt``.  All known
    texts are concatenated (separated by double newlines) into one side
    of the verification pair; the unknown text forms the other.

    Args:
        data_dir: Path to the root corpus directory containing the
                  per-problem subdirectories and ``truth.txt``.

    Returns:
        A list of records with ``text`` and ``label`` keys, emitted in
        pairs (known, unknown) per problem.
    """
    truth_path = os.path.join(data_dir, "truth.txt")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth.txt not found in {data_dir}")

    problem_labels: Dict[str, int] = {}
    with open(truth_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pid = parts[0]
                label_char = parts[1]
                problem_labels[pid] = 1 if label_char == "Y" else 0

    samples: List[Dict[str, Any]] = []
    for pid in sorted(problem_labels.keys()):
        problem_dir = os.path.join(data_dir, pid)
        if not os.path.exists(problem_dir):
            continue

        files = os.listdir(problem_dir)
        known_files = sorted(f for f in files if f.startswith("known"))
        unknown_files = [f for f in files if f.startswith("unknown")]

        if not known_files or not unknown_files:
            continue

        try:
            known_texts = []
            for kf in known_files:
                with open(os.path.join(problem_dir, kf), "r", encoding="utf-8", errors="replace") as f:
                    known_texts.append(f.read())
            text_a = "\n\n".join(known_texts)

            with open(os.path.join(problem_dir, unknown_files[0]), "r", encoding="utf-8", errors="replace") as f:
                text_b = f.read()
        except Exception:
            continue

        label_str = f"trial_{pid}_true" if problem_labels[pid] == 1 else f"trial_{pid}_false"

        samples.append({"text": text_a, "label": label_str})
        samples.append({"text": text_b, "label": label_str})

    return samples
