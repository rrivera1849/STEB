"""Extract overall results from summary sheet for Table 1 in the paper.

Reads the summary sheet, maps Excel model names to LaTeX display names,
and outputs LaTeX rows with bold best and underlined second-best per column.
Values are multiplied by 100 and shown as XX.YY.
"""
import argparse

import pandas as pd

SHEET_NAME = "summary"

# LaTeX display name -> Excel model name
# Models not in the Excel will show "--" for all cells.
MODEL_GROUPS = {
    "Style-specific": {
        "LUAR-MUD": "LUAR-MUD",
        "LUAR-CRUD": "LUAR-CRUD",
        "StyleDistance": "styledistance",
        "mStyleDistance": "mstyledistance",
        "StyleDistance (Synthetic)": "styledistance_synthetic_only",
        "multilingual-style-representation": "multilingual-style-representation",
        "Style-Embedding": "Style-Embedding",
        "STAR": "star",
        "LISA": "lisa_checkpoint",
    },
    "General-purpose sentence embedders": {
        "all-mpnet-base-v2": "all-mpnet-base-v2",
        "GTE-base-en-v1.5": "gte-base-en-v1.5",
        "GTE-large-en-v1.5": "gte-large-en-v1.5",
        "E5-base-v2": "e5-base-v2",
        "E5-large-v2": "e5-large-v2",
        "BGE-base-en-v1.5": "bge-base-en-v1.5",
        "BGE-large-en-v1.5": "bge-large-en-v1.5",
        "Jina Embeddings v3": "jina-embeddings-v3",
        "Qwen3-Embedding-8B": "Qwen3-Embedding-8B",
    },
    "Masked language models": {
        "BERT-large-cased": "bert-large-cased",
        "BERT-large-uncased": "bert-large-uncased",
        "RoBERTa-base": "roberta-base",
        "RoBERTa-large": "roberta-large",
        "DeBERTa-v3-base": "deberta-v3-base",
        "DeBERTa-v3-large": "deberta-v3-large",
        "ModernBERT-base": "ModernBERT-base",
        "ModernBERT-large": "ModernBERT-large",
    },
    "Causal language models": {
        "GPT-2 XL": "gpt2-xl",
        "OPT-1.3B": "opt-1.3b",
        "Qwen2-0.5B": "Qwen2-0.5B",
        "Qwen3-0.6B-Base": "Qwen3-0.6B-Base",
        "Qwen3.5-0.8B-Base": "Qwen3.5-0.8B-Base",
        "Qwen3.5-2B-Base": "Qwen3.5-2B-Base",
        "Qwen3.5-4B-Base": "Qwen3.5-4B-Base",
    },
    "Pre-defined Features": {
        "neurobiber": "neurobiber",
        "surface_pos": "surface_pos.yaml",
        "tfidfngrams_fineweb_sample10bt_1-2grams.pkl": "tfidfngrams_fineweb_sample10bt_1-2grams.pkl",
        "tfidfngrams_fineweb_sample10bt_1-3grams.pkl": "tfidfngrams_fineweb_sample10bt_1-3grams.pkl",
        "tfidfngrams_mud_subset_1-2grams.pkl": "tfidfngrams_mud_subset_1-2grams.pkl",
        "tfidfngrams_mud_subset_1-3grams.pkl": "tfidfngrams_mud_subset_1-3grams.pkl",
        "functionwordfreq": "functionwordfreq",
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
        help="Print every model ranked by 'STEB_score (avg)' descending, "
             "rows only.",
    )
    return parser.parse_args()


def format_row(
    display_name: str,
    excel_name: str,
    df: pd.DataFrame,
    best_per_col: dict,
    second_per_col: dict,
) -> str:
    """Render one LaTeX row for a model.

    Args:
        display_name: Name printed in the leftmost column.
        excel_name: Row index in the Excel sheet.
        df: The summary DataFrame.
        best_per_col: Per-column best value (scaled by 100).
        second_per_col: Per-column second-best value (scaled by 100, may be None).

    Returns:
        A LaTeX row string ending in `` \\\\``.
    """
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
    return f"{display_name} & {' & '.join(cells)} \\\\"


def main() -> None:
    """Entry point for Table 1 extraction."""
    args = parse_args()
    df = pd.read_excel(args.excel_path, sheet_name=SHEET_NAME, index_col=0)
    df = df.drop("# datasets", errors="ignore")

    excel_cols = [c for c, _ in COLUMNS]
    best_per_col = {}
    second_per_col = {}
    for col in excel_cols:
        valid = df[col].dropna().sort_values(ascending=False)
        best_val = valid.iloc[0]
        second_vals = valid[valid < best_val]
        second_val = second_vals.iloc[0] if len(second_vals) > 0 else None
        best_per_col[col] = round(best_val * 100, 2)
        second_per_col[col] = round(second_val * 100, 2) if second_val is not None else None

    excel_to_display = {
        excel: display
        for models in MODEL_GROUPS.values()
        for display, excel in models.items()
    }

    if args.all:
        rank_col = "STEB_score (avg)"
        models = df.sort_values(rank_col, ascending=False).index.tolist()
        for excel_name in models:
            display_name = excel_to_display.get(excel_name, excel_name)
            print(format_row(display_name, excel_name, df, best_per_col, second_per_col))
        return

    for _, models in MODEL_GROUPS.items():
        for display_name, excel_name in models.items():
            print(format_row(display_name, excel_name, df, best_per_col, second_per_col))


if __name__ == "__main__":
    main()
