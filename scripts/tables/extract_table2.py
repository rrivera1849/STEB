"""Extract top-5 plus style embeddings from summary_tasks for Table 2.

Reads the summary_tasks sheet, picks the top-5 models by the ``average``
column plus every style-specific model, and prints a LaTeX table body
with bold best and underlined second-best per column. The two groups
are separated by an italic header row. Authorship verification is
rendered as a multicolumn header over four sub-columns (Overall, Easy,
Medium, Hard). Values are multiplied by 100 and shown as XX.YY.
"""
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd

SHEET_NAME = "summary_tasks"

# LaTeX display name -> Excel model name, in the order they should be
# rendered when included as style-specific rows below the top-5.
STYLE_SPECIFIC = {
    "LUAR-MUD": "LUAR-MUD",
    "LUAR-CRUD": "LUAR-CRUD",
    "StyleDistance": "styledistance",
    "mStyleDistance": "mstyledistance",
    "StyleDistance (Synthetic)": "styledistance_synthetic_only",
    "multilingual-style-representation": "multilingual-style-representation",
    "Style-Embedding": "Style-Embedding",
    "STAR": "star",
    "LISA": "lisa_checkpoint",
    "NeuroBiber": "neurobiber",
}

# Column groups for the multicolumn header. Each group is
# (group_header, [(sub_header, excel_column), ...]). When a group has
# only one sub-column, sub_header is empty and the group header is used
# directly. When it has multiple, the group header spans the sub-cols
# via \multicolumn and a \cmidrule is drawn underneath.
COLUMN_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("AI-Text Det.",          [("", "machine_text_detection")]),
    ("AI-Text Det. (Adv.)",   [("", "machine_text_detection_adversarial")]),
    ("Authorship Verification", [
        ("Overall", "authorship_verification"),
        ("Easy",    "authorship_verification_easy"),
        ("Medium",  "authorship_verification_medium"),
        ("Hard",    "authorship_verification_hard"),
    ]),
    ("Authorship Retr.",      [("", "authorship_retrieval")]),
    ("Avg.",                  [("", "average")]),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "excel_path",
        help="Path to the STEB scores Excel workbook.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every model ranked by 'average' descending in one "
             "ready-to-paste LaTeX tabular, with no diagnostic output.",
    )
    return parser.parse_args()


def collect_columns(
    df: pd.DataFrame,
    quiet: bool = False,
) -> List[str]:
    """Return the ordered list of Excel column names rendered in the table.

    Skips any sub-column that is absent from ``df``; a warning is printed
    once per missing column so a user can spot YAML/Excel drift (unless
    ``quiet`` is set).

    Args:
        df: The summary_tasks DataFrame.
        quiet: If True, suppress the missing-column warnings.

    Returns:
        Flat list of Excel column names in display order.
    """
    cols: List[str] = []
    for _, subs in COLUMN_GROUPS:
        for _, excel_col in subs:
            if excel_col in df.columns:
                cols.append(excel_col)
            elif not quiet:
                print(f"  warning: column {excel_col!r} not in {SHEET_NAME}, skipping")
    return cols


def compute_best_second(
    df_100: pd.DataFrame,
    cols: List[str],
) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    """Compute per-column best and second-best values across all models.

    Args:
        df_100: The summary_tasks DataFrame, already scaled by 100.
        cols: Columns to consider.

    Returns:
        A pair of dicts (best_per_col, second_per_col), each keyed by
        column name. second_per_col entries are None when no second-best
        exists (e.g. a one-row column).
    """
    best: Dict[str, float] = {}
    second: Dict[str, Optional[float]] = {}
    for col in cols:
        valid = df_100[col].dropna().sort_values(ascending=False)
        if valid.empty:
            best[col] = float("nan")
            second[col] = None
            continue
        best_val = valid.iloc[0]
        below = valid[valid < best_val]
        best[col] = best_val
        second[col] = below.iloc[0] if len(below) > 0 else None
    return best, second


def render_header() -> str:
    """Render the two-row LaTeX header with multicolumn + cmidrule.

    Returns:
        Lines joined by newlines (no trailing newline) for printing.
    """
    has_multi = any(len(subs) > 1 for _, subs in COLUMN_GROUPS)

    # Row 1: group headers
    row1_cells = ["\\textbf{Model}"]
    cmidrules: List[str] = []
    col_offset = 2  # 1-indexed; col 1 is the model name
    for group_header, subs in COLUMN_GROUPS:
        n = len(subs)
        if n == 1:
            row1_cells.append(f"\\textbf{{{group_header}}}")
        else:
            row1_cells.append(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{group_header}}}}}")
            cmidrules.append(f"\\cmidrule(lr){{{col_offset}-{col_offset + n - 1}}}")
        col_offset += n
    row1 = " & ".join(row1_cells) + " \\\\"

    if not has_multi:
        return row1

    # Row 2: sub-headers (empty cells under single-column groups)
    row2_cells = [""]
    for _, subs in COLUMN_GROUPS:
        if len(subs) == 1:
            row2_cells.append("")
        else:
            row2_cells.extend(f"\\textbf{{{sub}}}" for sub, _ in subs)
    row2 = " & ".join(row2_cells) + " \\\\"

    return "\n".join([row1, " ".join(cmidrules), row2])


def format_row(
    display_name: str,
    excel_name: str,
    df_100: pd.DataFrame,
    cols: List[str],
    best: Dict[str, float],
    second: Dict[str, Optional[float]],
) -> str:
    """Render one LaTeX row for a model.

    Args:
        display_name: Name printed in the leftmost column.
        excel_name: Row index in the Excel sheet.
        df_100: The summary_tasks DataFrame, already scaled by 100.
        cols: Ordered list of column names to render.
        best: Per-column best value.
        second: Per-column second-best value (may be None).

    Returns:
        A LaTeX row string ending in `` \\\\``.
    """
    cells = []
    for col in cols:
        val = df_100.at[excel_name, col]
        if pd.isna(val):
            cells.append("--")
            continue
        cell = f"{val:.2f}"
        if val == best[col]:
            cell = f"\\textbf{{{cell}}}"
        elif second[col] is not None and val == second[col]:
            cell = f"\\underline{{{cell}}}"
        cells.append(cell)
    return f"{display_name} & {' & '.join(cells)} \\\\"


def main() -> None:
    """Entry point for Table 2 extraction."""
    args = parse_args()
    df = pd.read_excel(args.excel_path, sheet_name=SHEET_NAME, index_col=0)
    df_100 = (df * 100).round(2)

    cols = collect_columns(df, quiet=True)
    best, second = compute_best_second(df_100, cols)
    excel_to_display = {excel: display for display, excel in STYLE_SPECIFIC.items()}

    if args.all:
        models = df.sort_values("average", ascending=False).index.tolist()
    else:
        top5 = df.sort_values("average", ascending=False).head(5).index.tolist()
        remaining = [
            excel for _, excel in STYLE_SPECIFIC.items()
            if excel in df.index and excel not in top5
        ]
        models = top5 + remaining

    for excel_name in models:
        display_name = excel_to_display.get(excel_name, excel_name)
        print(format_row(display_name, excel_name, df_100, cols, best, second))


if __name__ == "__main__":
    main()
