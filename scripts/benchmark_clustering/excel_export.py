"""Export benchmark scores to a multi-sheet Excel workbook."""
from typing import Dict, List, Optional

import pandas as pd

from .config import OA_VARIANT_METRICS, TASK_METRICS
from .discovery import _warn_missing_metric, discover_all_scores


def export_excel(
    results_dir: str,
    output_path: str,
    include_excluded: bool = False,
    task_scores: Optional[Dict[str, pd.Series]] = None,
    task_metrics: Optional[Dict[str, str]] = None,
    manual_cluster_tables: Optional[Dict[str, pd.DataFrame]] = None,
    manual_cluster_datasets: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> None:
    """Export all scores to an Excel file.

    Sheets:
      - "scores": one row per (dataset, task, episode_config) with per-model
        metric values. Best per row is bold, second best is underlined.
      - "summary": cluster-aware aggregated scores per model per task
        (if --task or --all-tasks was used).
      - One sheet per cluster named "mc_{cluster}" with manual cluster
        averages (if --manual-clusters was used). Best per column is bold,
        second best is underlined. Includes dataset list per column.

    Args:
        results_dir: Path to the root results directory.
        output_path: Path for the output .xlsx file.
        include_excluded: If True, include semantic and non-English datasets.
        task_scores: Mapping from task name to per-model aggregated scores.
            If provided, written as the "summary" sheet.
        task_metrics: Mapping from task name to metric name (for column headers).
        manual_cluster_tables: Mapping from cluster name to DataFrame of manual
            cluster averages (models x tasks).
        manual_cluster_datasets: Mapping from cluster name to dict of column
            name to list of dataset names included in that column.
    """
    rows = discover_all_scores(results_dir, include_excluded)
    if not rows:
        print("No scores found. Nothing to export.")
        return

    records = []
    warned_missing: set = set()
    for (dataset, task, episode_config), model_metrics in sorted(rows.items()):
        # For order_alignment, emit one row per recognised variant metric so
        # readers can see both acc_mean and distractor_acc_mean side by side.
        # For all other tasks, emit a single row with TASK_METRICS' default.
        if task == "order_alignment":
            metric_keys = sorted(set(OA_VARIANT_METRICS.values()))
        else:
            metric_keys = [TASK_METRICS[task]]

        for primary_metric in metric_keys:
            record: Dict[str, object] = {
                "dataset": dataset,
                "task": task,
                "episode_config": episode_config,
                "primary_metric": primary_metric,
            }
            for model, metrics in model_metrics.items():
                value = metrics.get(primary_metric)
                if value is None:
                    _warn_missing_metric(dataset, task, primary_metric, warned_missing)
                    continue
                record[model] = value
            records.append(record)

    scores_df = pd.DataFrame(records)

    # Ensure metadata columns come first, then models sorted alphabetically
    meta_cols = ["dataset", "task", "episode_config", "primary_metric"]
    model_cols = sorted(c for c in scores_df.columns if c not in meta_cols)
    scores_df = scores_df[meta_cols + model_cols]

    from openpyxl.styles import Font

    bold_font = Font(bold=True)
    underline_font = Font(underline="single")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        scores_df.to_excel(writer, sheet_name="scores", index=False)

        # Bold best, underline second best per row (across model columns)
        ws_scores = writer.sheets["scores"]
        model_col_start = len(meta_cols) + 1  # 1-indexed, after meta columns
        model_col_end = model_col_start + len(model_cols) - 1
        for row_idx in range(2, len(scores_df) + 2):  # skip header
            vals = []
            for col_idx in range(model_col_start, model_col_end + 1):
                cell = ws_scores.cell(row=row_idx, column=col_idx)
                if isinstance(cell.value, (int, float)):
                    vals.append((cell.value, col_idx))
            if len(vals) < 2:
                continue
            vals.sort(key=lambda x: x[0], reverse=True)
            ws_scores.cell(row=row_idx, column=vals[0][1]).font = bold_font
            ws_scores.cell(row=row_idx, column=vals[1][1]).font = underline_font

        if task_scores and task_metrics:
            columns = {
                f"{task} ({task_metrics[task]})": scores
                for task, scores in task_scores.items()
            }
            summary_df = pd.DataFrame(columns)
            summary_df.index.name = "model"
            summary_df.to_excel(writer, sheet_name="summary")

            # Bold best, underline second best per column (across models)
            ws_summary = writer.sheets["summary"]
            for col_idx in range(2, len(summary_df.columns) + 2):  # skip index col
                vals = []
                for row_idx in range(2, len(summary_df) + 2):  # skip header
                    cell = ws_summary.cell(row=row_idx, column=col_idx)
                    if isinstance(cell.value, (int, float)):
                        vals.append((cell.value, row_idx))
                if len(vals) < 2:
                    continue
                vals.sort(key=lambda x: x[0], reverse=True)
                ws_summary.cell(row=vals[0][1], column=col_idx).font = bold_font
                ws_summary.cell(row=vals[1][1], column=col_idx).font = underline_font

        if manual_cluster_tables:
            col_ds_all = manual_cluster_datasets or {}
            for cluster_name, mc_df in sorted(manual_cluster_tables.items()):
                sheet_name = f"mc_{cluster_name}"[:31]  # Excel 31-char limit
                col_ds = col_ds_all.get(cluster_name, {})

                # Find max number of datasets across columns for row offset
                max_datasets = max(
                    (len(ds) for ds in col_ds.values()),
                    default=0,
                )
                # Leave rows for: "Datasets:" label + one row per dataset + blank row
                data_start_row = max_datasets + 2 if max_datasets > 0 else 0
                mc_df.to_excel(writer, sheet_name=sheet_name, startrow=data_start_row)

                ws = writer.sheets[sheet_name]

                # Write dataset lists above the data
                if col_ds:
                    italic_font = Font(italic=True)
                    ws.cell(row=1, column=1, value="Datasets:").font = italic_font
                    for col_idx, col_name in enumerate(mc_df.columns, start=2):
                        datasets = col_ds.get(col_name, [])
                        for ds_idx, ds_name in enumerate(datasets):
                            cell = ws.cell(row=1 + ds_idx, column=col_idx, value=ds_name)
                            cell.font = italic_font

                # Bold best, underline second best per column
                header_row = data_start_row + 1
                for col_idx in range(2, len(mc_df.columns) + 2):
                    vals = []
                    for row_idx in range(header_row + 1, header_row + len(mc_df) + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        if isinstance(cell.value, (int, float)):
                            vals.append((cell.value, row_idx))
                    if len(vals) < 2:
                        continue
                    vals.sort(key=lambda x: x[0], reverse=True)
                    ws.cell(row=vals[0][1], column=col_idx).font = bold_font
                    ws.cell(row=vals[1][1], column=col_idx).font = underline_font

        # Auto-resize columns for all sheets
        for ws in writer.book.worksheets:
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = max_len + 2

    n_sheets = 1
    if task_scores:
        n_sheets += 1
    if manual_cluster_tables:
        n_sheets += len(manual_cluster_tables)
    print(f"Exported {len(scores_df)} rows × {len(model_cols)} models ({n_sheets} sheets) to {output_path}")
