"""Build the public STEB leaderboard from a benchmark Excel workbook.

Reads ``scores.xlsx`` produced by ``python -m scripts.benchmark_clustering``
(specifically the ``STEB_operational`` and ``STEB_definitional`` sheets)
and emits a single Markdown page at ``docs/leaderboard.md`` with one
table per score. Numbers are scaled by 100 and rounded to 2 decimals to
match the format used in the paper.

The maintainer regenerates this file after refreshing ``scores.xlsx``;
the docs workflow then deploys it to GitHub Pages as part of the
mkdocs build.

Usage:
    python scripts/build_leaderboard.py [--excel PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EXCEL = _REPO_ROOT / "scores.xlsx"
_DEFAULT_OUTPUT = _REPO_ROOT / "docs" / "leaderboard.md"

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
        help=f"Path to the generated Markdown (default: {_DEFAULT_OUTPUT.relative_to(_REPO_ROOT)}).",
    )
    return parser.parse_args()


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


def _format_value(
    val: float,
    is_best: bool,
    is_second: bool,
) -> str:
    """Render a single score cell for the Markdown table.

    Args:
        val: The score value (already scaled to 0–100).
        is_best: Whether this cell holds the column's best score.
        is_second: Whether this cell holds the column's second-best score.

    Returns:
        A formatted string. Best is wrapped in ``**…**``, second is wrapped
        in ``*…*`` to stand out in rendered Markdown.
    """
    if pd.isna(val):
        return "—"
    s = f"{val:.2f}"
    if is_best:
        return f"**{s}**"
    if is_second:
        return f"*{s}*"
    return s


def _render_table(
    df: pd.DataFrame,
    sort_column: str,
    caption: str,
) -> str:
    """Render a model-table DataFrame as a sortable-ish Markdown table.

    The best per column is bolded; the second-best is italicized. Rows
    are sorted by ``sort_column`` descending.

    Args:
        df: Models × columns DataFrame with score values (already scaled
            by 100).
        sort_column: Column to sort rows by, descending.
        caption: Optional caption to render above the table.

    Returns:
        The Markdown table as a single string.
    """
    if sort_column not in df.columns:
        raise ValueError(
            f"Expected sort column {sort_column!r} not in DataFrame columns: "
            f"{list(df.columns)}"
        )

    df = df.sort_values(sort_column, ascending=False)

    best_per_col = {}
    second_per_col = {}
    for col in df.columns:
        valid = df[col].dropna().sort_values(ascending=False)
        if valid.empty:
            best_per_col[col] = None
            second_per_col[col] = None
            continue
        best_per_col[col] = valid.iloc[0]
        worse = valid[valid < valid.iloc[0]]
        second_per_col[col] = worse.iloc[0] if len(worse) > 0 else None

    lines: List[str] = []
    if caption:
        lines.append(caption)
        lines.append("")
    header_cells = ["Model"] + list(df.columns)
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for model, row in df.iterrows():
        cells = [str(model)]
        for col in df.columns:
            val = row[col]
            is_best = best_per_col[col] is not None and val == best_per_col[col]
            is_second = (
                second_per_col[col] is not None
                and val == second_per_col[col]
                and not is_best
            )
            cells.append(_format_value(val, is_best, is_second))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(
    excel_path: Path,
    output_path: Path,
) -> None:
    """Generate the leaderboard Markdown page from the benchmark workbook.

    Args:
        excel_path: Path to the benchmark Excel workbook.
        output_path: Where to write the Markdown.
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Benchmark workbook not found at {excel_path}. Run "
            f"`python -m scripts.benchmark_clustering` first."
        )

    op_df = pd.read_excel(excel_path, sheet_name=_OPERATIONAL_SHEET, index_col=0)
    op_df = _drop_metadata_rows(op_df) * 100

    def_df = pd.read_excel(excel_path, sheet_name=_DEFINITIONAL_SHEET, index_col=0)
    def_df = _drop_metadata_rows(def_df) * 100

    op_table = _render_table(
        op_df,
        _OPERATIONAL_SORT_COLUMN,
        caption="Sorted by `STEB_score (avg)` descending. **Bold** = best per column, *italic* = second best.",
    )
    def_table = _render_table(
        def_df,
        _DEFINITIONAL_SORT_COLUMN,
        caption="Sorted by `Definitional score` descending. **Bold** = best per column, *italic* = second best.",
    )

    md = f"""# Leaderboard

The canonical STEB leaderboard. Numbers reproduce the headline tables in the [STEB paper](https://github.com/rrivera1849/STEB/blob/main/STEB_paper.pdf) and are regenerated whenever the maintainer refreshes `scores.xlsx` from the latest benchmark runs.

There are two STEB scores, with different aggregation philosophies:

- **Operational** mirrors how the field has historically organized style work: macro-average within auto-discovered redundancy clusters per task, then across tasks. See `STEB_operational` below.
- **Definitional** scores embeddings against the style definition of Wegmann et al. (2026): the average of three axes — Object of Study (Genre, Register, Time, Demographics, Dialect, Idiolect), Linguistic Features, and Content Independence. See `STEB_definitional` below.

To submit a new model, see [`SUBMISSION.md`](https://github.com/rrivera1849/STEB/blob/main/SUBMISSION.md).

## STEB (Operational)

{op_table}

## STEB (Definitional)

{def_table}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    resolved = output_path.resolve()
    try:
        display_path = resolved.relative_to(_REPO_ROOT)
    except ValueError:
        display_path = resolved
    print(f"Wrote {display_path} "
          f"({len(op_df)} models x {len(op_df.columns)} operational cols, "
          f"{len(def_df.columns)} definitional cols).")


def main() -> None:
    """Entry point for the leaderboard builder."""
    args = parse_args()
    build(args.excel, args.output)


if __name__ == "__main__":
    main()
