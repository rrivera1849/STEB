"""Extract overall results from summary sheet for Table 1 in the paper.

Reads the summary sheet, maps Excel model names to LaTeX display names,
and outputs LaTeX rows with bold best and underlined second-best per column.
Values are multiplied by 100 and shown as XX.YY.
"""
import pandas as pd

EXCEL_PATH = "scores_20260506.xlsx"
SHEET_NAME = "summary"

# LaTeX display name -> Excel model name
# Models not in the Excel will show "--" for all cells.
MODEL_GROUPS = {
    "Style-specific": {
        "LUAR-MUD": "LUAR-MUD",
        "LUAR-CRUD": "LUAR-CRUD",
        "StyleDistance": "styledistance",
        "mStyleDistance": "mstyledistance",
        "multilingual-style-representation": "multilingual-style-representation",
        "Style-Embedding": "Style-Embedding",
        "STAR": "star",
    },
    "General-purpose sentence embedders": {
        "all-mpnet-base-v2": "all-mpnet-base-v2",
        "GTE-base-en-v1.5": "gte-base-en-v1.5",
        "GTE-large-en-v1.5": "gte-large-en-v1.5",
        "E5-base-v2": "e5-base-v2",
        "E5-large-v2": "e5-large-v2",
        "E5-Mistral-7B-Instruct": "e5-mistral-7b-instruct",
        "BGE-base-en-v1.5": "bge-base-en-v1.5",
        "BGE-large-en-v1.5": "bge-large-en-v1.5",
        "Jina Embeddings v3": "jina-embeddings-v3",
        "Qwen3-Embedding-8B": "Qwen3-Embedding-8B",
    },
    "Masked language models": {
        "RoBERTa-base": "roberta-base",
        "RoBERTa-large": "roberta-large",
        "DeBERTa-v3-base": None,
        "DeBERTa-v3-large": None,
    },
    "Causal language models": {
        "GPT-2 XL": "gpt2-xl",
        "Llama-3.2-1B": "Llama-3.2-1B",
        "Mistral-7B-v0.3": None,
    },
}

# Columns in the Excel -> short LaTeX header names
COLUMNS = [
    ("clustering (v_measure)", "Clust."),
    ("all_to_all_pair_classification (auc)", "A2A"),
    ("pre_defined_pair_classification (auc)", "PD"),
    ("order_alignment (distractor_acc_mean)", "Order Al."),
    ("retrieval (mrr)", "Retr."),
    ("probing (average)", "Probing"),
    ("STEB_score (avg)", "STEB"),
]


def main() -> None:
    """Entry point for Table 1 extraction."""
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, index_col=0)

    # Drop the # datasets row for scoring
    n_datasets_row = df.loc["# datasets"] if "# datasets" in df.index else None
    df = df.drop("# datasets", errors="ignore")

    excel_cols = [c for c, _ in COLUMNS]

    # Compute best and second-best per column across all models
    best_per_col = {}
    second_per_col = {}
    for col in excel_cols:
        valid = df[col].dropna().sort_values(ascending=False)
        best_val = valid.iloc[0]
        second_vals = valid[valid < best_val]
        second_val = second_vals.iloc[0] if len(second_vals) > 0 else None
        best_per_col[col] = round(best_val * 100, 2)
        second_per_col[col] = round(second_val * 100, 2) if second_val is not None else None

    print("Best and second-best per column (x100):")
    for col, short in COLUMNS:
        s = second_per_col[col]
        print(f"  {short}: best={best_per_col[col]}, second={s}")
    print()

    # Print # datasets row
    if n_datasets_row is not None:
        ds_vals = [str(int(n_datasets_row[c])) if pd.notna(n_datasets_row[c]) else "--" for c, _ in COLUMNS]
        print(f"\\# datasets & {' & '.join(ds_vals)} \\\\")
    print()

    # Print LaTeX rows grouped
    n_cols = len(COLUMNS) + 1  # +1 for model name
    for group_name, models in MODEL_GROUPS.items():
        print(f"\\midrule")
        print(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        for display_name, excel_name in models.items():
            if excel_name is None or excel_name not in df.index:
                cells = ["--"] * len(COLUMNS)
            else:
                cells = []
                for col, _ in COLUMNS:
                    raw = df.at[excel_name, col]
                    if pd.isna(raw):
                        cells.append("--")
                        continue
                    val = round(raw * 100, 2)
                    val_str = f"{val:.2f}"
                    if val == best_per_col[col]:
                        val_str = f"\\textbf{{{val_str}}}"
                    elif second_per_col[col] is not None and val == second_per_col[col]:
                        val_str = f"\\underline{{{val_str}}}"
                    cells.append(val_str)
            print(f"{display_name} & {' & '.join(cells)} \\\\")


if __name__ == "__main__":
    main()
