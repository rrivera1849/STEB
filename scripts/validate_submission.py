"""Validate STEB community submissions.

Checks the contents of ``SUBMISSIONS.yaml``, ``submitted_results/``, and
``scripts/models_all.txt`` for the kinds of mistakes a first-time
contributor is most likely to make. Designed to run both locally
(before opening a PR) and in CI on every PR that touches those paths.

Exit code 0 = clean. Non-zero = at least one failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

REQUIRED_KEYS: Tuple[str, ...] = ("short_name", "hf_id", "run_command", "contributor")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMISSIONS_YAML = _REPO_ROOT / "SUBMISSIONS.yaml"
_SUBMITTED_RESULTS_DIR = _REPO_ROOT / "submitted_results"
_MODELS_FILE = _REPO_ROOT / "scripts" / "models_all.txt"


def load_submissions() -> List[Dict[str, Any]]:
    """Parse SUBMISSIONS.yaml and return its list of entries.

    Returns:
        A list of dicts, one per submission entry. An empty list if the
        file is empty or has no entries.

    Raises:
        FileNotFoundError: If SUBMISSIONS.yaml is missing.
        ValueError: If the top-level YAML is not a list.
    """
    if not _SUBMISSIONS_YAML.exists():
        raise FileNotFoundError(f"Missing {_SUBMISSIONS_YAML}")
    with open(_SUBMISSIONS_YAML) as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(
            f"{_SUBMISSIONS_YAML.name} must contain a top-level YAML list, "
            f"got {type(data).__name__}."
        )
    return data


def collect_short_names_in_results() -> Set[str]:
    """Return the set of model short-names present under submitted_results/.

    A model is considered present if at least one
    ``submitted_results/<dataset>/<short_name>/`` directory exists. The
    function does not validate the inner episode-config / task / metrics
    structure here; that's covered by ``check_results_json_valid``.

    Returns:
        Set of short_name strings observed on disk.
    """
    short_names: Set[str] = set()
    if not _SUBMITTED_RESULTS_DIR.exists():
        return short_names
    for dataset_dir in _SUBMITTED_RESULTS_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        for model_dir in dataset_dir.iterdir():
            if model_dir.is_dir():
                short_names.add(model_dir.name)
    return short_names


def collect_models_in_models_file() -> Set[str]:
    """Return the set of model identifiers listed in scripts/models_all.txt.

    Each non-blank, non-comment line is split on '/' and both the full
    line and the short_name (the part after the last '/') are added to
    the set so callers can match either form.

    Returns:
        Set of identifier strings.
    """
    out: Set[str] = set()
    if not _MODELS_FILE.exists():
        return out
    with open(_MODELS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line)
            out.add(line.split("/")[-1])
    return out


def check_entry_schema(
    entry: Any,
    idx: int,
    errors: List[str],
) -> bool:
    """Verify a single SUBMISSIONS.yaml entry has the required keys.

    Args:
        entry: The entry to validate. Expected to be a dict.
        idx: Zero-indexed position in the SUBMISSIONS.yaml list, used in
            error messages.
        errors: Mutable list that errors are appended to.

    Returns:
        True if the entry is structurally valid, False otherwise.
    """
    if not isinstance(entry, dict):
        errors.append(f"Entry #{idx}: must be a mapping, got {type(entry).__name__}.")
        return False
    ok = True
    for key in REQUIRED_KEYS:
        if key not in entry:
            errors.append(f"Entry #{idx}: missing required key '{key}'.")
            ok = False
            continue
        if not isinstance(entry[key], str) or not entry[key].strip():
            errors.append(
                f"Entry #{idx} ('{entry.get('short_name', '?')}'): "
                f"key '{key}' must be a non-empty string."
            )
            ok = False
    return ok


def check_results_json_valid(errors: List[str]) -> None:
    """Walk submitted_results/ and verify every metrics.json parses as JSON.

    Args:
        errors: Mutable list that errors are appended to.
    """
    if not _SUBMITTED_RESULTS_DIR.exists():
        return
    for metrics_file in _SUBMITTED_RESULTS_DIR.rglob("metrics.json"):
        try:
            with open(metrics_file) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            errors.append(
                f"Malformed JSON in {metrics_file.relative_to(_REPO_ROOT)}: {e}"
            )


def validate() -> int:
    """Run all submission checks and print a summary.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        entries = load_submissions()
    except (FileNotFoundError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    short_names_on_disk = collect_short_names_in_results()
    models_in_file = collect_models_in_models_file()

    declared_short_names: Set[str] = set()
    for idx, entry in enumerate(entries):
        if not check_entry_schema(entry, idx, errors):
            continue

        short_name = entry["short_name"]
        declared_short_names.add(short_name)

        if short_name not in short_names_on_disk:
            errors.append(
                f"Entry '{short_name}': no matching directory under "
                f"submitted_results/<dataset>/{short_name}/. Did you forget "
                f"to copy your results subtree?"
            )

        if (
            entry["hf_id"] not in models_in_file
            and short_name not in models_in_file
        ):
            errors.append(
                f"Entry '{short_name}': neither '{entry['hf_id']}' nor "
                f"'{short_name}' is listed in scripts/models_all.txt. Add "
                f"the model so it's picked up by the default --models-file."
            )

    # Orphans on disk: submitted_results entries with no SUBMISSIONS.yaml row.
    for orphan in sorted(short_names_on_disk - declared_short_names):
        warnings.append(
            f"submitted_results/ contains '{orphan}' but no entry in "
            f"SUBMISSIONS.yaml mentions it."
        )

    check_results_json_valid(errors)

    print(f"Submission validation — {len(entries)} entries declared, "
          f"{len(short_names_on_disk)} short_names on disk.")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\nErrors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(f"\nFAIL — {len(errors)} error(s).", file=sys.stderr)
        return 1

    print("\nOK — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
