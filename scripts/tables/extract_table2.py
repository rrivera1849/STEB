"""Extract top-5 models from summary_tasks sheet for Table 2 in the paper.

Values are multiplied by 100 and shown as XX.YY.
"""
import pandas as pd

EXCEL_PATH = "scores_20260506.xlsx"
SHEET_NAME = "summary_tasks"


def main() -> None:
    """Entry point for Table 2 extraction."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, index_col=0)

    # Sort by average column, descending
    df_sorted = df.sort_values("average", ascending=False)

    # Pick top 5
    top5 = df_sorted.head(5)

    print("Top 5 models by average score:")
    print(top5.to_string())
    print()

    # Multiply by 100 for the paper (keep 2 decimal places)
    top5_100 = (top5 * 100).round(2)
    df_100 = (df * 100).round(2)

    print("Scaled (x100):")
    print(top5_100.to_string())
    print()

    # Determine best and second-best per column (across ALL models, not just top 5)
    best_per_col = {}
    second_per_col = {}
    for col in df.columns:
        valid = df_100[col].dropna().sort_values(ascending=False)
        best_val = valid.iloc[0]
        second_vals = valid[valid < best_val]
        second_val = second_vals.iloc[0] if len(second_vals) > 0 else None
        best_per_col[col] = best_val
        second_per_col[col] = second_val

    print("Best and second-best per column (across all models):")
    for col in df.columns:
        s = second_per_col[col]
        print(f"  {col}: best={best_per_col[col]}, second={s}")
    print()

    print("LaTeX rows:")
    for model in top5.index:
        cells = []
        for col in df.columns:
            val = top5_100.at[model, col]
            if pd.isna(val):
                cells.append("--")
                continue
            val_str = f"{val:.2f}"
            if val == best_per_col[col]:
                val_str = f"\\textbf{{{val_str}}}"
            elif second_per_col[col] is not None and val == second_per_col[col]:
                val_str = f"\\underline{{{val_str}}}"
            cells.append(val_str)
        print(f"{model} & {' & '.join(cells)} \\\\")


if __name__ == "__main__":
    main()
