"""
Runs every canonical STEB model on the datasets that don't have results yet.

Intended for the "we just added a dataset, now backfill the leaderboard"
workflow: it diffs the datasets STEB knows about against what's already in
the results tree, splits the canonical models evenly across the GPUs on the
machine, and runs each GPU's model chunk against every new dataset.

Resuming a partial run is handled by ``steb`` itself: ``steb.core.evaluate``
already skips a (dataset, model, task) combination when its ``metrics.json``
exists, so re-invoking this script after a kill/crash just fills in the
gaps. Nothing here re-implements that check.

Usage:
    python -m scripts.run_new_datasets
    python -m scripts.run_new_datasets --debug
    python -m scripts.run_new_datasets --datasets authbench_attribution_en authbench_attribution_ja
    python -m scripts.run_new_datasets --dry-run
"""

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Optional, Tuple

MODELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_all.txt")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

DEBUG_MAX_MODELS = 2
DEBUG_MAX_DATASETS = 1

# "dummy_*" are test-only fixtures with no real leaderboard entry, and
# "fisher_*" requires licensed LDC audio transcripts most contributors don't
# have, so neither ever gets a results directory. Excluded from
# auto-discovery so they don't get treated as "new" on every run.
EXCLUDED_DATASET_PREFIXES = ("dummy_", "fisher_")


def parse_models_file(
    path: str,
) -> List[Tuple[str, str]]:
    """
    Parses a models file into (short_name, model_name_or_path) pairs.

    The short name is the same string ``steb.core.evaluate`` uses to name a
    model's results directory (``os.path.basename`` of the model string), so
    it can be used to check whether a model already has results.

    Args:
        path: Path to a plain-text file, one model per line. Blank lines and
            lines starting with "#" are ignored.

    Returns:
        A list of (short_name, model_name_or_path) tuples, in file order.
    """
    models = []
    with open(path) as f:
        for line in f:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            short_name = os.path.basename(entry.rstrip("/"))
            models.append((short_name, entry))
    return models


def discover_new_datasets(
    all_datasets: List[str],
    results_dir: str,
    exclude_prefixes: Tuple[str, ...] = EXCLUDED_DATASET_PREFIXES,
) -> List[str]:
    """
    Finds datasets that have no results directory yet.

    Args:
        all_datasets: Every dataset name STEB knows about.
        results_dir: Root of the results tree (e.g. ``./results``).
        exclude_prefixes: Dataset name prefixes to skip regardless of
            whether they have results (e.g. test-only or license-gated
            datasets that never get real results).

    Returns:
        The subset of ``all_datasets`` with no matching subdirectory under
        ``results_dir`` and no excluded prefix, in the same order they were
        given.
    """
    return [
        dataset_name
        for dataset_name in all_datasets
        if not dataset_name.startswith(exclude_prefixes)
        and not os.path.isdir(os.path.join(results_dir, os.path.basename(dataset_name)))
    ]


def discover_gpus() -> List[int]:
    """
    Lists the GPU indices visible to this machine via ``nvidia-smi``.

    Returns:
        A list of GPU indices, e.g. ``[0, 1, 2, 3]``. Empty if ``nvidia-smi``
        isn't available or reports no devices.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def partition_models(
    models: List[Tuple[str, str]],
    num_workers: int,
) -> List[List[Tuple[str, str]]]:
    """
    Splits models evenly (by count, not by size/cost) across workers.

    Args:
        models: The (short_name, model_name_or_path) pairs to split.
        num_workers: Number of chunks to produce, one per GPU.

    Returns:
        A list of ``num_workers`` chunks, each a list of model pairs. Chunk
        sizes differ by at most one model.
    """
    base_size, remainder = divmod(len(models), num_workers)
    chunks = []
    start = 0
    for i in range(num_workers):
        chunk_size = base_size + (1 if i < remainder else 0)
        chunks.append(models[start:start + chunk_size])
        start += chunk_size
    return chunks


def build_command(
    model_name_or_path: str,
    dataset_name: str,
    results_dir: str,
) -> List[str]:
    """
    Builds the ``steb`` invocation for one (model, dataset) pair.

    Uses the ``all`` subcommand so every task configured for the dataset
    runs, regardless of the dataset's task type.

    Args:
        model_name_or_path: The model to evaluate.
        dataset_name: The dataset to evaluate it on.
        results_dir: Where ``steb`` should write ``metrics.json`` files.

    Returns:
        The argv list to pass to ``subprocess.run``.
    """
    return [
        "steb", "all", model_name_or_path,
        "--dataset", dataset_name,
        "--output-folder", results_dir,
        "--progress-bar",
    ]


def run_worker(
    gpu_id: Optional[int],
    models: List[Tuple[str, str]],
    datasets: List[str],
    results_dir: str,
    log_path: str,
) -> None:
    """
    Runs one GPU worker's assigned models against every new dataset.

    Every command's stdout/stderr is appended to ``log_path`` so the run can
    be watched with ``tail -f``. ``steb`` itself skips (dataset, model, task)
    combinations that already have a ``metrics.json``, so a killed and
    restarted worker just resumes where it left off.

    Args:
        gpu_id: CUDA device index to pin this worker to, or ``None`` to run
            without setting ``CUDA_VISIBLE_DEVICES`` (e.g. CPU-only models).
        models: The (short_name, model_name_or_path) pairs this worker owns.
        datasets: The new dataset names to evaluate every model on.
        results_dir: Where ``steb`` should write results.
        log_path: File to append this worker's subprocess output to.
    """
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(log_path, "a") as log_file:
        for short_name, model_name_or_path in models:
            for dataset_name in datasets:
                command = build_command(model_name_or_path, dataset_name, results_dir)
                header = f"\n=== [gpu {gpu_id}] {short_name} | {dataset_name} | {' '.join(command)} ===\n"
                log_file.write(header)
                log_file.flush()
                subprocess.run(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-file", default=MODELS_FILE, help="Plain-text file of canonical model IDs.")
    parser.add_argument("--results-dir", default=None, help="Results directory (default: steb's configured RESULTS_DIR).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Evaluate exactly these datasets instead of auto-detecting new ones.")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU indices to use instead of auto-detecting via nvidia-smi.")
    parser.add_argument("--log-dir", default=LOG_DIR, help="Directory for per-GPU log files.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running anything.")
    parser.add_argument("--debug", action="store_true", help=f"Limit to the first {DEBUG_MAX_MODELS} models and {DEBUG_MAX_DATASETS} dataset(s), with extra logging.")
    return parser.parse_args()


def main() -> None:
    """Discovers new datasets and canonical models, then dispatches evaluation across GPUs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    from steb import get_all_datasets
    from steb.utils import RESULTS_DIR

    results_dir = args.results_dir or RESULTS_DIR
    models = parse_models_file(args.models_file)
    datasets = args.datasets or discover_new_datasets(get_all_datasets(), results_dir)

    if args.debug:
        models = models[:DEBUG_MAX_MODELS]
        datasets = datasets[:DEBUG_MAX_DATASETS]
        logging.info("Debug mode: %d model(s), %d dataset(s)", len(models), len(datasets))

    if not datasets:
        logging.info("No new datasets found under %s. Nothing to do.", results_dir)
        return
    if not models:
        logging.info("No canonical models found in %s. Nothing to do.", args.models_file)
        return

    logging.info("New datasets (%d): %s", len(datasets), datasets)
    logging.info("Canonical models (%d): %s", len(models), [m[0] for m in models])

    gpus = args.gpus if args.gpus is not None else discover_gpus()
    num_workers = len(gpus) if gpus else 1
    chunks = partition_models(models, num_workers)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_new_datasets_{run_id}")

    worker_specs = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        gpu_id = gpus[i] if gpus else None
        log_path = os.path.join(log_dir, f"gpu{gpu_id if gpu_id is not None else i}.log")
        worker_specs.append((gpu_id, chunk, log_path))
        logging.info(
            "GPU %s -> %d model(s): %s (log: %s)",
            gpu_id, len(chunk), [m[0] for m in chunk], log_path,
        )

    if args.dry_run:
        logging.info("Dry run, not launching any workers.")
        return

    os.makedirs(log_dir, exist_ok=True)
    logging.info("Logs: tail -f %s/*.log", log_dir)
    with ProcessPoolExecutor(max_workers=len(worker_specs)) as executor:
        futures = [
            executor.submit(run_worker, gpu_id, chunk, datasets, results_dir, log_path)
            for gpu_id, chunk, log_path in worker_specs
        ]
        for future in futures:
            future.result()

    logging.info("Done. Rebuild the leaderboard with: python -m scripts.benchmark_clustering")


if __name__ == "__main__":
    main()
