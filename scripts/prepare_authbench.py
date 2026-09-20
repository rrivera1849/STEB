"""
Prepares a per-language subset of AuthBench's test split for STEB's
retrieval task.

Downloads the ``queries``, ``candidates``, and ``ground_truth`` configs of
the ``test`` split of the ``MaoXun/AuthBench`` dataset from Hugging Face
once, filters to one (or every) language, joins each query to its author
(via ground_truth), and writes JSONL records compatible with
``steb.loaders.retrieval.default_retrieval_loader`` /
``default_retrieval_record_handler`` -- one STEB dataset per language,
named ``authbench_attribution_<lang>``.

It also attaches per-record metadata (genre, primary_genre, length_bucket)
that default_retrieval_record_handler passes through automatically, and
writes predicate-based "submetrics" specs into each dataset's config.json
(see steb/core.py:_matches_submetric_predicate), so a single dataset per
language can report length-bucket and topic-controlled retrieval
breakdowns without needing separate datasets per bucket/genre, and without
inlining large label lists.

Usage:
    python scripts/prepare_authbench.py --lang en [--debug] [--log-file PATH]
    python scripts/prepare_authbench.py --all-langs

Debug mode (--debug) restricts to a small number of queries/candidates and
prints verbose per-record information, per project convention.
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_DATASET_PATH = "MaoXun/AuthBench"

# All languages AuthBench's test split covers.
AUTHBENCH_LANGUAGES: List[str] = [
    "en", "ru", "zh", "ar", "de", "es", "ko", "fr", "ja", "hi",
]

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


def dataset_name(lang: str) -> str:
    """
    Returns the STEB dataset name for a given AuthBench language code.

    Args:
        lang: An AuthBench language code (e.g. "en", "ja").

    Returns:
        The corresponding STEB dataset name, e.g. "authbench_attribution_en".
    """
    return f"authbench_attribution_{lang}"


def raw_data_dir(lang: str) -> str:
    """Default raw data directory for a language's dataset."""
    return os.path.join(REPO_ROOT, "raw_datasets", dataset_name(lang))


def config_path(lang: str) -> str:
    """Path to a language's dataset config.json."""
    return os.path.join(REPO_ROOT, "steb", "steb_datasets", dataset_name(lang), "config.json")


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
    submetrics predicates; the metadata is otherwise ignored at load time.

    Queries without a resolved author (i.e. missing from ground_truth) are
    dropped, since they cannot be matched to any candidate.

    Args:
        query_rows: Language-filtered rows from the queries config.
        candidate_rows: Language-filtered rows from the candidates config.
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


def build_submetrics_config(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Builds predicate-based submetrics for length-bucket and topic-controlled
    (primary-genre) retrieval breakdowns (see
    steb.core._matches_submetric_predicate).

    Length-bucket submetrics use "sides": "query" -- only queries in the
    target bucket are scored, but every candidate stays in the pool
    (matching AuthBench's own per-bucket evaluation, which restricts which
    queries are scored while keeping the full candidate pool).

    Primary-genre (topic-controlled) submetrics use "sides": "both" -- both
    queries and candidates are restricted to the same genre, which is what
    limits the candidate pool to same-topic documents. The set of genres is
    read off the actual records (not hardcoded), so this stays correct
    per-language even though genre distributions vary a lot by language.

    Args:
        records: The full list of retrieval records for one language (only
            used to read off the set of primary genres present).

    Returns:
        A dict mapping submetric name to its predicate spec, ready to go
        straight into config.json's "submetrics".
    """
    submetrics: Dict[str, Dict[str, Any]] = {
        f"length_{bucket_name}": {
            "field": "length_bucket", "value": bucket_name, "sides": "query",
        }
        for bucket_name, _, _ in LENGTH_BUCKETS
    }
    genres = sorted({r["primary_genre"] for r in records})
    for genre in genres:
        submetrics[f"topic_pool_{genre}"] = {
            "field": "primary_genre", "value": genre, "sides": "both",
        }
    return submetrics


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


def build_config(lang: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds the config.json contents for one language's AuthBench dataset.

    Submetrics are predicate specs (see build_submetrics_config), not
    literal label lists, so they're small enough to inline directly --
    no generated sidecar file needed.

    Args:
        lang: The AuthBench language code this dataset covers.
        records: The full list of retrieval records, passed through to
            build_submetrics_config() to read off the set of genres.

    Returns:
        The config dict, ready to be JSON-serialized.
    """
    name = dataset_name(lang)
    return {
        "dataset_name": name,
        "type": "custom",
        "data_dir": name,
        "loader_module": "steb.loaders.retrieval",
        "loader_function": "default_retrieval_loader",
        "record_handler": {
            "text_getter": "text",
            "label_getter": "label",
            "custom_record_handler_function": "default_retrieval_record_handler",
        },
        "tasks": {
            "retrieval": {
                "submetrics": build_submetrics_config(records),
            }
        },
    }


def fetch_authbench_test_split(logger: logging.Logger):
    """
    Downloads the full (all-language) queries/candidates/ground_truth
    configs of AuthBench's test split via the `datasets` library, once.

    Args:
        logger: Logger for progress messages.

    Returns:
        A tuple of (queries_ds, candidates_ds, ground_truth_ds), the raw
        (unfiltered) HuggingFace datasets.
    """
    from datasets import load_dataset

    logger.info("Downloading AuthBench test split from %s ...", HF_DATASET_PATH)
    queries_ds = load_dataset(HF_DATASET_PATH, "queries", split="test")
    candidates_ds = load_dataset(HF_DATASET_PATH, "candidates", split="test")
    ground_truth_ds = load_dataset(HF_DATASET_PATH, "ground_truth", split="test")
    logger.info(
        "Full test split: %d queries, %d candidates, %d ground_truth rows (all languages)",
        len(queries_ds), len(candidates_ds), len(ground_truth_ds),
    )
    return queries_ds, candidates_ds, ground_truth_ds


def filter_to_language(queries_ds, candidates_ds, ground_truth_ds, lang: str, debug: bool, logger: logging.Logger):
    """
    Filters the full test split down to one language.

    Args:
        queries_ds: The raw (unfiltered) queries dataset.
        candidates_ds: The raw (unfiltered) candidates dataset.
        ground_truth_ds: The raw (unfiltered) ground_truth dataset.
        lang: The AuthBench language code to filter to.
        debug: If True, truncates to a small number of rows for a fast,
            verbose dry run.
        logger: Logger for progress messages.

    Returns:
        A tuple of (query_rows, candidate_rows, ground_truth_rows), each a
        list of dicts, filtered to this language.
    """
    query_rows = [r for r in queries_ds if r["lang"] == lang]
    candidate_rows = [r for r in candidates_ds if r["lang"] == lang]

    query_ids = {r["query_id"] for r in query_rows}
    ground_truth_rows = [r for r in ground_truth_ds if r["query_id"] in query_ids]

    logger.info(
        "[%s] test split: %d queries, %d candidates, %d ground_truth rows",
        lang, len(query_rows), len(candidate_rows), len(ground_truth_rows),
    )

    if debug:
        query_rows = query_rows[:DEBUG_N_QUERIES]
        candidate_rows = candidate_rows[:DEBUG_N_CANDIDATES]
        kept_qids = {r["query_id"] for r in query_rows}
        ground_truth_rows = [r for r in ground_truth_rows if r["query_id"] in kept_qids]
        logger.info(
            "[%s] [debug] truncated to %d queries, %d candidates, %d ground_truth rows",
            lang, len(query_rows), len(candidate_rows), len(ground_truth_rows),
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
    logger = logging.getLogger("prepare_authbench")
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


def prepare_one_language(
    lang: str,
    queries_ds,
    candidates_ds,
    ground_truth_ds,
    logger: logging.Logger,
    debug: bool,
    out_raw_data_dir: str,
    out_config_path: Optional[str],
) -> None:
    """
    Builds and writes one language's authbench_attribution_<lang> dataset.

    Args:
        lang: The AuthBench language code to prepare.
        queries_ds: The raw (unfiltered) queries dataset.
        candidates_ds: The raw (unfiltered) candidates dataset.
        ground_truth_ds: The raw (unfiltered) ground_truth dataset.
        logger: Logger for progress messages.
        debug: If True, only logs what would happen; writes nothing.
        out_raw_data_dir: Where to write the raw JSONL.
        out_config_path: Where to write config.json, or None to skip
            writing it (e.g. when the download script only needs to
            refresh the raw JSONL, not the committed config).
    """
    query_rows, candidate_rows, ground_truth_rows = filter_to_language(
        queries_ds, candidates_ds, ground_truth_ds, lang, debug, logger,
    )

    query_author = build_query_author_map(ground_truth_rows)

    records = build_retrieval_records(
        tqdm(query_rows, desc=f"{lang} queries"),
        tqdm(candidate_rows, desc=f"{lang} candidates"),
        query_author,
    )
    n_queries = sum(1 for r in records if r["is_query"])
    n_candidates = len(records) - n_queries
    logger.info("[%s] built %d records (%d queries, %d candidates)", lang, len(records), n_queries, n_candidates)

    submetrics_config = build_submetrics_config(records)
    logger.info("[%s] computed %d predicate-based submetrics", lang, len(submetrics_config))

    if debug:
        logger.info("[%s] [debug] skipping writes to raw_datasets/ and config.json", lang)
        logger.info("[%s] [debug] example record: %s", lang, records[0] if records else None)
        return

    out_path = os.path.join(out_raw_data_dir, f"{dataset_name(lang)}.jsonl")
    write_jsonl(records, out_path)
    logger.info("[%s] wrote %d records to %s", lang, len(records), out_path)

    if out_config_path is None:
        logger.info("[%s] skipping config.json", lang)
        return

    config = build_config(lang, records)
    os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
    with open(out_config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    logger.info("[%s] wrote config to %s", lang, out_config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lang",
        choices=AUTHBENCH_LANGUAGES,
        default="en",
        help="AuthBench language code to prepare (default: en). Ignored if --all-langs is set.",
    )
    parser.add_argument(
        "--all-langs",
        action="store_true",
        help=f"Prepare every AuthBench language in one run: {AUTHBENCH_LANGUAGES}.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a small subset with verbose per-record logging.",
    )
    parser.add_argument(
        "--log-file",
        default=os.path.join(REPO_ROOT, "logs", "prepare_authbench.log"),
        help="Path to the log file (tail -f this to watch progress).",
    )
    parser.add_argument(
        "--raw-data-dir",
        default=None,
        help=(
            "Where to write the raw JSONL for a single --lang run (default: "
            "raw_datasets/authbench_attribution_<lang> under the repo root). "
            "Used by download_datasets.sh to respect a custom output directory. "
            "Not used with --all-langs (each language gets its own default dir)."
        ),
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help=(
            "Only (re)write the raw JSONL, not the committed config.json. Each "
            "language's config.json's submetrics are generated once from "
            "AuthBench's frozen test split and shouldn't need to be regenerated "
            "on every download_datasets.sh run."
        ),
    )
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info("Log file: %s", args.log_file)

    queries_ds, candidates_ds, ground_truth_ds = fetch_authbench_test_split(logger)

    langs = AUTHBENCH_LANGUAGES if args.all_langs else [args.lang]
    for lang in langs:
        out_raw_data_dir = (
            raw_data_dir(lang) if args.raw_data_dir is None or args.all_langs else args.raw_data_dir
        )
        out_config_path = None if args.skip_config else config_path(lang)
        prepare_one_language(
            lang, queries_ds, candidates_ds, ground_truth_ds, logger, args.debug,
            out_raw_data_dir, out_config_path,
        )


if __name__ == "__main__":
    main()
