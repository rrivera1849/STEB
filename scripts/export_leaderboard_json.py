"""Export the STEB leaderboard from a benchmark Excel workbook to JSON.

Reads ``scores.xlsx`` produced by ``python -m scripts.benchmark_clustering``
(the ``STEB_operational`` and ``STEB_definitional`` sheets) and emits a
single JSON file consumed by the static leaderboard page hosted on the
Hugging Face Space. This mirrors ``scripts/build_leaderboard.py`` (which
targets the GitHub Pages Markdown leaderboard), just with JSON output
instead of Markdown.

Usage:
    python scripts/export_leaderboard_json.py [--excel PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from run_new_datasets import parse_models_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EXCEL = _REPO_ROOT / "scores.xlsx"
_DEFAULT_OUTPUT = _REPO_ROOT / "leaderboard_data.json"
_DEFAULT_MODELS_FILE = _REPO_ROOT / "scripts" / "models_all.txt"

_OPERATIONAL_SHEET = "STEB_operational"
_OPERATIONAL_SORT_COLUMN = "STEB_score (avg)"
_DEFINITIONAL_SHEET = "STEB_definitional"
_DEFINITIONAL_SORT_COLUMN = "Definitional score"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--excel",
        type=Path,
        default=_DEFAULT_EXCEL,
        help=f"Path to scores.xlsx (default: {_DEFAULT_EXCEL.relative_to(_REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Path to the generated JSON (default: {_DEFAULT_OUTPUT.relative_to(_REPO_ROOT)}).",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=_DEFAULT_MODELS_FILE,
        help=f"Plain-text file of canonical model IDs (default: {_DEFAULT_MODELS_FILE.relative_to(_REPO_ROOT)}).",
    )
    return parser.parse_args()


def build_model_url_map(models_file: Path) -> Dict[str, str]:
    """Maps a model's short name to its Hugging Face Hub URL, where known.

    Only entries that look like a Hub repo id (``owner/name``, as opposed
    to a local checkpoint name or a pre-defined-feature file) get a URL.

    Args:
        models_file: Path to a ``run_new_datasets.parse_models_file``
            compatible model list, e.g. ``scripts/models_all.txt``.

    Returns:
        A dict from short model name (as used in ``scores.xlsx`` rows) to
        its ``https://huggingface.co/<owner>/<name>`` URL.
    """
    url_by_short_name = {}
    for short_name, model_name_or_path in parse_models_file(str(models_file)):
        if short_name != model_name_or_path and "/" in model_name_or_path:
            url_by_short_name[short_name] = f"https://huggingface.co/{model_name_or_path}"
    return url_by_short_name


def _drop_metadata_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the leading '# datasets' bookkeeping row from a sheet.

    Args:
        df: A DataFrame loaded from a benchmark_clustering sheet whose
            first row may be the '# datasets' summary.

    Returns:
        The same DataFrame with the '# datasets' row removed if present.
    """
    if "# datasets" in df.index:
        return df.drop("# datasets")
    return df


def _sheet_to_table(
    df: pd.DataFrame,
    sort_column: str,
    model_url_by_name: Dict[str, str],
) -> Dict[str, Any]:
    """Convert a models x columns DataFrame into a JSON-friendly table.

    Args:
        df: Models x columns DataFrame with score values (already scaled
            by 100).
        sort_column: Column to sort rows by, descending.
        model_url_by_name: Maps a model's short name to its Hugging Face
            Hub URL, where known (see ``build_model_url_map``).

    Returns:
        A dict with ``columns``, ``sort_column``, and ``rows`` (each row a
        dict of column name to value, plus ``"model"`` and
        ``"model_url"``, the latter ``None`` when the model has no known
        Hub checkpoint). NaNs become ``None`` so the JSON encodes them as
        ``null``.
    """
    df = df.sort_values(sort_column, ascending=False)
    columns = list(df.columns)
    rows = []
    for model, row in df.iterrows():
        record: Dict[str, Any] = {
            "model": str(model),
            "model_url": model_url_by_name.get(str(model)),
        }
        for col in columns:
            val = row[col]
            record[col] = None if pd.isna(val) else round(float(val), 2)
        rows.append(record)
    return {"sort_column": sort_column, "columns": columns, "rows": rows}


def export(
    excel_path: Path,
    output_path: Path,
    models_file: Path,
) -> None:
    """Generate the leaderboard JSON from the benchmark workbook.

    Args:
        excel_path: Path to the benchmark Excel workbook.
        output_path: Where to write the JSON.
        models_file: Plain-text file of canonical model IDs, used to link
            model names to their Hugging Face Hub checkpoints.
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Benchmark workbook not found at {excel_path}. Run "
            f"`python -m scripts.benchmark_clustering` first."
        )

    model_url_by_name = build_model_url_map(models_file)

    op_df = pd.read_excel(excel_path, sheet_name=_OPERATIONAL_SHEET, index_col=0)
    op_df = _drop_metadata_rows(op_df) * 100

    def_df = pd.read_excel(excel_path, sheet_name=_DEFINITIONAL_SHEET, index_col=0)
    def_df = _drop_metadata_rows(def_df) * 100

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operational": _sheet_to_table(op_df, _OPERATIONAL_SORT_COLUMN, model_url_by_name),
        "definitional": _sheet_to_table(def_df, _DEFINITIONAL_SORT_COLUMN, model_url_by_name),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(
        f"Wrote {output_path} "
        f"({len(op_df)} models x {len(op_df.columns)} operational cols, "
        f"{len(def_df.columns)} definitional cols)."
    )


def main() -> None:
    """Entry point for the leaderboard JSON exporter."""
    args = parse_args()
    export(args.excel, args.output, args.models_file)


if __name__ == "__main__":
    main()
