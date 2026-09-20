"""
Prepares the English subset of AuthBench's test split for STEB's retrieval task.

Downloads the ``queries``, ``candidates``, and ``ground_truth`` configs of the
``test`` split of the ``MaoXun/AuthBench`` dataset from Hugging Face, filters
to English documents, joins each query to its author (via ground_truth), and
writes JSONL records compatible with
``steb.loaders.retrieval.default_retrieval_loader`` /
``default_retrieval_record_handler``.

It also computes two sets of "submetrics" label lists (length-bucket and
primary-genre) and writes them into the dataset's config.json, so that a
single `authbench_attribution_en` dataset can report length-bucket and
topic-controlled retrieval breakdowns via STEB's existing submetrics
mechanism (see steb/core.py:_evaluate_submetrics), without needing separate
datasets per bucket/genre.

Usage:
    python scripts/prepare_authbench_en.py [--debug] [--log-file PATH]

Debug mode (--debug) restricts to a small number of queries/candidates and
prints verbose per-record information, per project convention.
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(REPO_ROOT, "raw_datasets", "authbench_attribution_en")
CONFIG_PATH = os.path.join(
    REPO_ROOT, "steb", "steb_datasets", "authbench_attribution_en", "config.json"
)

HF_DATASET_PATH = "MaoXun/AuthBench"
LANG = "en"
DEBUG_N_QUERIES = 20
DEBUG_N_CANDIDATES = 20

# Token-length bucket thresholds, as defined in the AuthBench paper
# (Sec. on document-length buckets): short 1-10, medium 11-100,
# long 101-500, extra-long >500 tokens.
LENGTH_BUCKETS: List[Tuple[str, int, int]] = [
    ("short", 1, 10),
    ("medium", 11, 100),
    ("long", 101, 500),
    ("extra_long", 501, float("inf")),
]


def length_bucket(token_length: int) -> str:
    """
    Maps a token length to one of AuthBench's four length buckets.

    Args:
        token_length: Number of tokens in the document.

    Returns:
        One of "short", "medium", "long", "extra_long".
    """
    for name, low, high in LENGTH_BUCKETS:
        if low <= token_length <= high:
            return name
    raise ValueError(f"token_length {token_length} did not match any bucket")


def primary_genre(genre: str) -> str:
    """
    Extracts the primary genre from a (possibly fine-grained) genre string.

    AuthBench genres look like "blog/student" or "news"; the primary genre
    is the segment before the first "/".

    Args:
        genre: The raw genre string.

    Returns:
        The primary genre.
    """
    return genre.split("/", 1)[0]


def build_query_author_map(ground_truth_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Builds a mapping from query_id to author_id from ground_truth rows.

    Args:
        ground_truth_rows: Rows from the ground_truth config, each with
            "query_id" and "author_id".

    Returns:
        Dict mapping query_id -> author_id.
    """
    return {row["query_id"]: row["author_id"] for row in ground_truth_rows}


def build_retrieval_records(
    query_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    query_author: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Converts raw AuthBench query/candidate rows into STEB retrieval records.

    Each output record has the fields STEB's default retrieval loader
    expects ("text", "label", "is_query"), plus extra metadata
    ("genre", "primary_genre", "length_bucket") used only to build
    submetrics label lists; the metadata is otherwise ignored at load time.

    Queries without a resolved author (i.e. missing from ground_truth) are
    dropped, since they cannot be matched to any candidate.

    Args:
        query_rows: English-filtered rows from the queries config.
        candidate_rows: English-filtered rows from the candidates config.
        query_author: Mapping from query_id to author_id.

    Returns:
        A list of retrieval records.
    """
    records = []

    for row in query_rows:
        author_id = query_author.get(row["query_id"])
        if author_id is None:
            continue
        records.append({
            "text": row["content"],
            "label": author_id,
            "is_query": True,
            "genre": row["genre"],
            "primary_genre": primary_genre(row["genre"]),
            "length_bucket": length_bucket(row["token_length"]),
        })

    for row in candidate_rows:
        records.append({
            "text": row["content"],
            "label": row["author_id"],
            "is_query": False,
            "genre": row["genre"],
            "primary_genre": primary_genre(row["genre"]),
            "length_bucket": length_bucket(row["token_length"]),
        })

    return records


def build_submetrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Builds submetrics label lists for length-bucket and topic-controlled
    (primary-genre) retrieval breakdowns.

    Length-bucket submetrics are asymmetric: only queries in the target
    bucket are included, but every candidate is kept, so retrieval is
    still scored against the full candidate pool (matching AuthBench's own
    per-bucket evaluation, which restricts which queries are scored while
    keeping the full candidate pool).

    Primary-genre (topic-controlled) submetrics are symmetric: both queries
    and candidates are restricted to the same genre, which is what limits
    the candidate pool to same-topic documents.

    Args:
        records: The full list of retrieval records, each with "label",
            "is_query", "primary_genre", and "length_bucket".

    Returns:
        A dict with two keys, "length_bucket" and "topic_pool", each
        mapping a submetric name to a list of labels to keep.
    """
    def suffixed_label(record: Dict[str, Any]) -> str:
        suffix = "_query" if record["is_query"] else "_target"
        return f"{record['label']}{suffix}"

    all_target_labels = [
        suffixed_label(r) for r in records if not r["is_query"]
    ]

    length_bucket_submetrics: Dict[str, List[str]] = {}
    for bucket_name, _, _ in LENGTH_BUCKETS:
        bucket_query_labels = [
            suffixed_label(r) for r in records
            if r["is_query"] and r["length_bucket"] == bucket_name
        ]
        length_bucket_submetrics[bucket_name] = bucket_query_labels + all_target_labels

    genres = sorted({r["primary_genre"] for r in records})
    topic_pool_submetrics: Dict[str, List[str]] = {}
    for genre in genres:
        genre_labels = [
            suffixed_label(r) for r in records if r["primary_genre"] == genre
        ]
        topic_pool_submetrics[genre] = genre_labels

    return {
        "length_bucket": length_bucket_submetrics,
        "topic_pool": topic_pool_submetrics,
    }


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """
    Writes records to a JSONL file, creating parent directories as needed.

    Args:
        records: The records to write, one per line.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


SUBMETRICS_FILENAME = "submetrics.json"


def flatten_submetrics(submetrics: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """
    Flattens build_submetrics()'s nested output into the flat
    {submetric_name: label_list} shape STEB's submetrics mechanism expects.

    Args:
        submetrics: Output of build_submetrics().

    Returns:
        A single dict mapping submetric name (e.g. "length_short",
        "topic_pool_news") to its label list.
    """
    return {
        **{
            f"length_{name}": labels
            for name, labels in submetrics["length_bucket"].items()
        },
        **{
            f"topic_pool_{genre}": labels
            for genre, labels in submetrics["topic_pool"].items()
        },
    }


def build_config() -> Dict[str, Any]:
    """
    Builds the config.json contents for authbench_attribution_en.

    The submetrics label lists are not inlined here: with ~9.7k
    author-ID-keyed labels per submetric, inlining them would make
    config.json unreviewable. Instead "submetrics" is a filename, resolved
    by steb.core._resolve_submetrics_config relative to the dataset's raw
    data directory at eval time (see write_submetrics_file()).

    Returns:
        The config dict, ready to be JSON-serialized.
    """
    return {
        "dataset_name": "authbench_attribution_en",
        "type": "custom",
        "data_dir": "authbench_attribution_en",
        "loader_module": "steb.loaders.retrieval",
        "loader_function": "default_retrieval_loader",
        "record_handler": {
            "text_getter": "text",
            "label_getter": "label",
            "custom_record_handler_function": "default_retrieval_record_handler",
        },
        "tasks": {
            "retrieval": {
                "submetrics": SUBMETRICS_FILENAME,
            }
        },
    }


def write_submetrics_file(submetrics: Dict[str, Dict[str, List[str]]], raw_data_dir: str) -> str:
    """
    Writes the flattened submetrics label lists to a JSON file alongside
    the raw JSONL, so config.json can reference it by filename instead of
    inlining it.

    Args:
        submetrics: Output of build_submetrics().
        raw_data_dir: The dataset's raw data directory.

    Returns:
        The path the submetrics file was written to.
    """
    path = os.path.join(raw_data_dir, SUBMETRICS_FILENAME)
    os.makedirs(raw_data_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(flatten_submetrics(submetrics), f)
    return path


def fetch_authbench_test_split(debug: bool, logger: logging.Logger):
    """
    Downloads the queries/candidates/ground_truth configs of AuthBench's
    test split via the `datasets` library.

    Args:
        debug: If True, truncates to a small number of rows for a fast,
            verbose dry run.
        logger: Logger for progress messages.

    Returns:
        A tuple of (query_rows, candidate_rows, ground_truth_rows), each a
        list of dicts, filtered to English.
    """
    from datasets import load_dataset

    logger.info("Downloading AuthBench test split from %s ...", HF_DATASET_PATH)
    queries_ds = load_dataset(HF_DATASET_PATH, "queries", split="test")
    candidates_ds = load_dataset(HF_DATASET_PATH, "candidates", split="test")
    ground_truth_ds = load_dataset(HF_DATASET_PATH, "ground_truth", split="test")

    query_rows = [r for r in queries_ds if r["lang"] == LANG]
    candidate_rows = [r for r in candidates_ds if r["lang"] == LANG]

    query_ids = {r["query_id"] for r in query_rows}
    ground_truth_rows = [r for r in ground_truth_ds if r["query_id"] in query_ids]

    logger.info(
        "English test split: %d queries, %d candidates, %d ground_truth rows",
        len(query_rows), len(candidate_rows), len(ground_truth_rows),
    )

    if debug:
        query_rows = query_rows[:DEBUG_N_QUERIES]
        candidate_rows = candidate_rows[:DEBUG_N_CANDIDATES]
        kept_qids = {r["query_id"] for r in query_rows}
        ground_truth_rows = [r for r in ground_truth_rows if r["query_id"] in kept_qids]
        logger.info(
            "[debug] truncated to %d queries, %d candidates, %d ground_truth rows",
            len(query_rows), len(candidate_rows), len(ground_truth_rows),
        )
        for row in query_rows:
            logger.debug("  query %s | genre=%s | token_length=%s", row["query_id"], row["genre"], row["token_length"])

    return query_rows, candidate_rows, ground_truth_rows


def setup_logger(log_file: str) -> logging.Logger:
    """
    Configures a logger that writes to both stdout and a log file.

    Args:
        log_file: Path to the log file (created if missing).

    Returns:
        A configured logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("prepare_authbench_en")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a small subset with verbose per-record logging.",
    )
    parser.add_argument(
        "--log-file",
        default=os.path.join(REPO_ROOT, "logs", "prepare_authbench_en.log"),
        help="Path to the log file (tail -f this to watch progress).",
    )
    parser.add_argument(
        "--raw-data-dir",
        default=RAW_DATA_DIR,
        help=(
            "Where to write the raw JSONL (default: raw_datasets/authbench_attribution_en "
            "under the repo root). Used by download_datasets.sh to respect a custom "
            "output directory."
        ),
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help=(
            "Only (re)write the raw JSONL, not steb/steb_datasets/authbench_attribution_en/"
            "config.json. The committed config.json's submetrics were generated once from "
            "AuthBench's frozen test split and shouldn't need to be regenerated on every "
            "download_datasets.sh run."
        ),
    )
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info("Log file: %s", args.log_file)

    query_rows, candidate_rows, ground_truth_rows = fetch_authbench_test_split(
        debug=args.debug, logger=logger,
    )

    query_author = build_query_author_map(ground_truth_rows)

    logger.info("Building retrieval records ...")
    records = build_retrieval_records(
        tqdm(query_rows, desc="queries"),
        tqdm(candidate_rows, desc="candidates"),
        query_author,
    )
    n_queries = sum(1 for r in records if r["is_query"])
    n_candidates = len(records) - n_queries
    logger.info("Built %d records (%d queries, %d candidates)", len(records), n_queries, n_candidates)

    submetrics = build_submetrics(records)
    logger.info(
        "Computed submetrics: %d length buckets, %d primary genres",
        len(submetrics["length_bucket"]), len(submetrics["topic_pool"]),
    )

    if args.debug:
        logger.info("[debug] skipping writes to raw_datasets/ and config.json")
        logger.info("[debug] example record: %s", records[0] if records else None)
        return

    out_path = os.path.join(args.raw_data_dir, "authbench_attribution_en.jsonl")
    write_jsonl(records, out_path)
    logger.info("Wrote %d records to %s", len(records), out_path)

    submetrics_path = write_submetrics_file(submetrics, args.raw_data_dir)
    logger.info("Wrote submetrics to %s", submetrics_path)

    if args.skip_config:
        logger.info("--skip-config set: leaving %s untouched", CONFIG_PATH)
        return

    config = build_config()
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    logger.info("Wrote config to %s", CONFIG_PATH)


if __name__ == "__main__":
    main()
