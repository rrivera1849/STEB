"""Extract top-5 models from summary_features sheet for Table 3 in the paper."""
import pandas as pd

EXCEL_PATH = "scores_20260506.xlsx"
SHEET_NAME = "summary_features"

df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, index_col=0)

# Sort by average column, descending
df_sorted = df.sort_values("average", ascending=False)

# Pick top 5
top5 = df_sorted.head(5)

print("Top 5 models by average score:")
print(top5.to_string())
print()

# Multiply by 100 for the paper
top5_100 = (top5 * 100).round(0).astype(int)

print("Scaled (x100):")
print(top5_100.to_string())
print()

# Determine best and second-best per column (across ALL models, not just top 5)
df_100 = (df * 100).round(0)

print("Best and second-best per column (across all models):")
for col in df.columns:
    valid = df_100[col].dropna().sort_values(ascending=False)
    best = valid.iloc[0]
    second = valid[valid < best].iloc[0] if len(valid[valid < best]) > 0 else None
    print(f"  {col}: best={best:.0f}, second={second:.0f}" if second else f"  {col}: best={best:.0f}, no second")
print()

# Get best/second per column across all models
best_per_col = {}
second_per_col = {}
for col in df.columns:
    valid = df_100[col].dropna().sort_values(ascending=False)
    best_val = valid.iloc[0]
    second_val = valid[valid < best_val].iloc[0] if len(valid[valid < best_val]) > 0 else None
    best_per_col[col] = best_val
    second_per_col[col] = second_val

print("LaTeX rows:")
for model in top5.index:
    cells = []
    for col in df.columns:
        val = top5_100.at[model, col]
        if pd.isna(val):
            cells.append("--")
        elif val == best_per_col[col]:
            cells.append(f"\\textbf{{{val}}}")
        elif second_per_col[col] is not None and val == second_per_col[col]:
            cells.append(f"\\underline{{{val}}}")
        else:
            cells.append(str(val))
    print(f"{model} & {' & '.join(cells)} \\\\")
