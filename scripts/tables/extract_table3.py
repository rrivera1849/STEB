"""Extract top-5 models from summary_features sheet for Table 3 in the paper.

Values are multiplied by 100 and shown as XX.YY.
"""
import argparse

import pandas as pd

SHEET_NAME = "summary_features"


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


def format_row(
    model: str,
    df_100: pd.DataFrame,
    cols: list,
    best_per_col: dict,
    second_per_col: dict,
) -> str:
    """Render one LaTeX row for a model.

    Args:
        model: Model name (Excel row index).
        df_100: Per-model DataFrame scaled by 100.
        cols: Ordered list of column names to render.
        best_per_col: Per-column best value.
        second_per_col: Per-column second-best value (may be None).

    Returns:
        A LaTeX row string ending in `` \\\\``.
    """
    cells = []
    for col in cols:
        val = df_100.at[model, col]
        if pd.isna(val):
            cells.append("--")
            continue
        val_str = f"{val:.2f}"
        if val == best_per_col[col]:
            val_str = f"\\textbf{{{val_str}}}"
        elif second_per_col[col] is not None and val == second_per_col[col]:
            val_str = f"\\underline{{{val_str}}}"
        cells.append(val_str)
    return f"{model} & {' & '.join(cells)} \\\\"


def main() -> None:
    """Entry point for Table 3 extraction."""
    args = parse_args()
    df = pd.read_excel(args.excel_path, sheet_name=SHEET_NAME, index_col=0)
    df_100 = (df * 100).round(2)
    cols = list(df.columns)

    # Determine best and second-best per column (across ALL models, not just top 5)
    best_per_col = {}
    second_per_col = {}
    for col in cols:
        valid = df_100[col].dropna().sort_values(ascending=False)
        best_val = valid.iloc[0]
        second_vals = valid[valid < best_val]
        second_val = second_vals.iloc[0] if len(second_vals) > 0 else None
        best_per_col[col] = best_val
        second_per_col[col] = second_val

    if args.all:
        models = df.sort_values("average", ascending=False).index.tolist()
    else:
        models = df.sort_values("average", ascending=False).head(5).index.tolist()

    for model in models:
        print(format_row(model, df_100, cols, best_per_col, second_per_col))


if __name__ == "__main__":
    main()
