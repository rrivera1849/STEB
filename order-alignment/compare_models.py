#!/usr/bin/env python3
"""
Compare model responses across CSV files from multiple folders.

Usage:
    python compare_models.py folder1 folder2 -o output.md
    python compare_models.py /path/to/results1 /path/to/results2 -o comparison.md
"""

import pandas as pd
import argparse
from pathlib import Path


def create_comparison_markdown(folders: list[str], output_path: str):
    """Read CSV files from folders and create a markdown comparison."""

    # Collect all CSV files from the folders
    csv_files = []
    for folder in folders:
        folder_path = Path(folder)
        if folder_path.is_file() and folder_path.suffix == '.csv':
            csv_files.append(folder_path)
        elif folder_path.is_dir():
            csv_files.extend(folder_path.glob('*.csv'))
        else:
            print(f"Warning: {folder} is not a valid file or directory")

    if not csv_files:
        print("No CSV files found!")
        return

    print(f"Found {len(csv_files)} CSV files\n")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        model_name = df['model'].iloc[0] if 'model' in df.columns else f.stem
        short_name = model_name.split('/')[-1].replace('_', '-')
        df['short_model'] = short_name
        dfs.append(df)
        print(f"  {short_name}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    unique_inputs = combined['input'].unique()
    models = sorted(combined['short_model'].unique())

    print(f"\nFound {len(unique_inputs)} unique inputs across {len(models)} models")

    # Create markdown
    md = "# Model Response Comparison\n\n"

    for inp in unique_inputs:
        md += f"## Input: `{inp}`\n\n"
        subset = combined[combined['input'] == inp]

        for model in models:
            model_data = subset[subset['short_model'] == model]
            if len(model_data) == 0:
                continue
            row = model_data.iloc[0]

            md += f"### {model}\n\n"
            md += "| Level | Response |\n"
            md += "|-------|----------|\n"

            for level in ['level1', 'level2', 'level3', 'level4']:
                if level not in row:
                    continue
                val = row[level]
                if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
                    val = '*(empty)*'
                else:
                    val = str(val).replace('|', '\\|').replace('\n', ' ').replace('\r', '')
                md += f"| {level} | {val} |\n"

            md += "\n"

        md += "---\n\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare model responses from CSV files in folders')
    parser.add_argument('folders', nargs='+', help='Folders containing CSV files (or individual CSV files)')
    parser.add_argument('-o', '--output', default='model_comparison.md',
                        help='Output markdown file (default: model_comparison.md)')

    args = parser.parse_args()

    create_comparison_markdown(args.folders, args.output)


if __name__ == '__main__':
    main()
    # python compare_models.py graded_closed_models/gradually-increase graded_closed_model_outputs/gradually-increase -o comparison.md