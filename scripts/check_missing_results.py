"""Check for missing metrics across models and report errors from logs.

Scans the results directory to find which (dataset, task) combinations
each model is missing, then looks up the error from the most recent log
file for that model.

Usage:
    python scripts/check_missing_results.py [--results-dir results] [--output missing_results.md]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from steb.core import get_supported_datasets
from steb.utils import RESULTS_DIR

from scripts.benchmark_clustering.config import EXCLUDED_DATASETS, EXCLUDED_MODELS, TASK_METRICS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Check for missing metrics and report errors from logs.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Path to the results directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="missing_results.md",
        help="Path to save the markdown report (default: %(default)s).",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include datasets that are excluded by default.",
    )
    return parser.parse_args()


def discover_models(
    results_dir: Path,
) -> Set[str]:
    """Find all model names present in the results directory.

    Args:
        results_dir: Path to the root results directory.

    Returns:
        A set of model directory names.
    """
    models: Set[str] = set()
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == "logs":
            continue
        for model_dir in dataset_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in EXCLUDED_MODELS:
                models.add(model_dir.name)
    return models


def discover_existing_results(
    results_dir: Path,
    include_excluded: bool = False,
) -> Dict[str, Set[Tuple[str, str]]]:
    """Find all (dataset, task) pairs each model has results for.

    Args:
        results_dir: Path to the root results directory.
        include_excluded: If True, include excluded datasets.

    Returns:
        A dict mapping model name to a set of (dataset, task) tuples.
    """
    existing: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "logs":
            continue
        if not include_excluded and dataset_dir.name in EXCLUDED_DATASETS:
            continue

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name in EXCLUDED_MODELS:
                continue

            for ep_dir in model_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                for task_dir in ep_dir.iterdir():
                    if not task_dir.is_dir():
                        continue
                    metrics_file = task_dir / "metrics.json"
                    if metrics_file.exists():
                        existing[model_dir.name].add((dataset_dir.name, task_dir.name))

    return existing


def build_expected_results(
    include_excluded: bool = False,
) -> Set[Tuple[str, str]]:
    """Build the set of all expected (dataset, task) pairs.

    Args:
        include_excluded: If True, include excluded datasets.

    Returns:
        A set of (dataset, task) tuples.
    """
    expected: Set[Tuple[str, str]] = set()
    for task_name in TASK_METRICS:
        for dataset in get_supported_datasets(task_name):
            if not include_excluded and dataset in EXCLUDED_DATASETS:
                continue
            expected.add((dataset, task_name))
    return expected


def load_all_failures(
    results_dir: Path,
) -> Dict[str, List[dict]]:
    """Load failures from all log files, keeping the most recent per (model, dataset, task).

    Args:
        results_dir: Path to the root results directory.

    Returns:
        A dict mapping model name to a list of failure dicts.
    """
    logs_dir = results_dir / "logs"
    if not logs_dir.exists():
        return {}

    # Collect (model, log_path, timestamp) triples
    log_entries: List[Tuple[str, Path, str]] = []
    for log_file in sorted(logs_dir.glob("*.log.json")):
        if "_benchmark_" not in log_file.stem:
            continue
        model_name, timestamp = log_file.stem.rsplit("_benchmark_", 1)
        log_entries.append((model_name, log_file, timestamp))

    # Process oldest first so newer entries overwrite older ones
    log_entries.sort(key=lambda x: x[2])

    # Key: (model, dataset, task) -> failure dict
    failure_index: Dict[Tuple[str, str, str], dict] = {}
    for model_name, log_path, _ in log_entries:
        with open(log_path) as f:
            log = json.load(f)
        for fail in log.get("failures", []):
            key = (model_name, fail["dataset"], fail["task"])
            failure_index[key] = fail

    # Group by model
    failures: Dict[str, List[dict]] = defaultdict(list)
    for (model_name, _, _), fail in failure_index.items():
        failures[model_name].append(fail)

    return dict(failures)


def _normalize_error(
    error: str,
) -> str:
    """Collapse an error message into a canonical key for grouping.

    Strips paths and dataset-specific names so that errors differing
    only in the dataset or path are grouped together.

    Args:
        error: The raw error string from a log file.

    Returns:
        A shortened, normalized error string.
    """
    import re

    normalized = re.sub(r"/\S+", "<path>", error)
    normalized = re.sub(r"dataset: \S+", "dataset: <name>", normalized)
    normalized = re.sub(r"'[^']*'", "'<name>'", normalized)
    normalized = re.sub(r"[\d.]+ [KMGT]iB", "<size>", normalized)
    return normalized.strip()


def build_report(
    models: Set[str],
    expected: Set[Tuple[str, str]],
    existing: Dict[str, Set[Tuple[str, str]]],
    failures: Dict[str, List[dict]],
) -> str:
    """Build a markdown report of missing results grouped by error.

    Args:
        models: Set of all model names.
        expected: Set of all expected (dataset, task) pairs.
        existing: Dict mapping model to set of (dataset, task) it has.
        failures: Dict mapping model to list of failure dicts from logs.

    Returns:
        A markdown-formatted report string.
    """
    # Index failures by (model, dataset, task) for lookup
    failure_index: Dict[Tuple[str, str, str], str] = {}
    for model, fails in failures.items():
        for fail in fails:
            key = (model, fail["dataset"], fail["task"])
            failure_index[key] = fail.get("error", "unknown")

    # Collect all missing (model, dataset, task) with their error
    missing_entries: List[Tuple[str, str, str, str]] = []
    for model in sorted(models):
        model_existing = existing.get(model, set())
        for dataset, task in sorted(expected - model_existing):
            error = failure_index.get((model, dataset, task), "")
            missing_entries.append((model, dataset, task, error))

    # Group by normalized error
    error_groups: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
    for model, dataset, task, error in missing_entries:
        key = _normalize_error(error) if error else ""
        error_groups[key].append((model, dataset, task, error))

    lines = [
        "# Missing Results Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nModels: {len(models)} | Expected (dataset, task) pairs: {len(expected)} | Total missing: {len(missing_entries)}",
        "",
    ]

    if not missing_entries:
        lines.append("All models have complete results.")
        return "\n".join(lines)

    # Never-attempted first
    never_attempted = error_groups.pop("", [])
    if never_attempted:
        lines.append(f"## Never Attempted ({len(never_attempted)} missing)")
        lines.append("")
        lines.append("| Model | Dataset | Task |")
        lines.append("|-------|---------|------|")
        for model, dataset, task, _ in never_attempted:
            lines.append(f"| {model} | {dataset} | {task} |")
        lines.append("")

    # Then each error group
    for i, (norm_error, entries) in enumerate(sorted(error_groups.items()), 1):
        example_error = entries[0][3]
        lines.append(f"## Error Group {i} ({len(entries)} missing)")
        lines.append("")
        lines.append(f"**Error:** `{example_error[:200]}`")
        lines.append("")
        lines.append("| Model | Dataset | Task |")
        lines.append("|-------|---------|------|")
        for model, dataset, task, _ in entries:
            lines.append(f"| {model} | {dataset} | {task} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Entry point for missing results checker."""
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    models = discover_models(results_dir)
    expected = build_expected_results(args.include_excluded)
    existing = discover_existing_results(results_dir, args.include_excluded)
    failures = load_all_failures(results_dir)

    report = build_report(models, expected, existing, failures)

    print(report)

    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()
