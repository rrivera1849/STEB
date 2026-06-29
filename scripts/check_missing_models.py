"""Check which models are missing results for a given dataset.

Compares models listed in a models file against the subdirectories
present in the results directory for a specific dataset.
"""
import argparse
from pathlib import Path


DEFAULT_RESULTS_DIR = "results"
DEFAULT_MODELS_FILE = "scripts/models_all.txt"


def parse_models_file(
    models_file: str,
) -> list[str]:
    """Parse a models file and return the short model names.

    Each line is expected to be either a comment (starting with #),
    blank, or a model identifier like 'org/model-name'. The short
    name is the part after the last '/'.

    Args:
        models_file: Path to the models file.

    Returns:
        List of short model names.
    """
    models = []
    with open(models_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            short_name = line.split("/")[-1]
            models.append(short_name)
    return models


def check_missing(
    results_dir: str,
    dataset: str,
    models: list[str],
) -> list[str]:
    """Return model names that have no results for the given dataset.

    Args:
        results_dir: Path to the root results directory.
        dataset: Name of the dataset to check.
        models: List of short model names to look for.

    Returns:
        List of model names missing from the dataset's results.
    """
    dataset_path = Path(results_dir) / dataset
    if not dataset_path.is_dir():
        return models

    present = {d.name for d in dataset_path.iterdir() if d.is_dir()}
    return [m for m in models if m not in present]


def main() -> None:
    """Entry point for the missing-models checker."""
    parser = argparse.ArgumentParser(
        description="Check which models are missing results for a dataset.",
    )
    parser.add_argument(
        "dataset",
        help="Name of the dataset to check.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to the results directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--models-file",
        default=DEFAULT_MODELS_FILE,
        help=f"Path to the models file (default: {DEFAULT_MODELS_FILE}).",
    )
    args = parser.parse_args()

    models = parse_models_file(args.models_file)
    missing = check_missing(args.results_dir, args.dataset, models)

    if not missing:
        print(f"All {len(models)} models have results for '{args.dataset}'.")
    else:
        print(f"Missing {len(missing)}/{len(models)} models for '{args.dataset}':")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
