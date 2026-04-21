# Probing

This directory contains the scripts used to create the probing datasets.

## Pipeline

The probing pipeline has two steps:

1. **Feature extraction** (`create_probing_datasets.py`): Extracts foundation-level
   linguistic features from raw text using [LFTK](https://github.com/brucewlee/lftk).

2. **Binning and splitting** (`split_probing_datasets.py`): Bins each feature into
   quantile labels, balances the dataset across bins, and assigns per-feature
   train/val/test splits. Outputs a single unified JSONL file with per-feature
   `label_*` and `split_*` fields.

## Usage

**Step 1** -- Extract features from a local JSONL file:

```bash
python scripts/probing/create_probing_datasets.py \
    --input_path /path/to/input.jsonl \
    --output_path /path/to/features.jsonl \
    --text_column text \
    --num_samples 40000
```

Or from a HuggingFace dataset:

```bash
python scripts/probing/create_probing_datasets.py \
    --hf_dataset AnnaWegmann/StyleEmbeddingData \
    --hf_split test \
    --text_column "Anchor (A)" \
    --output_path /path/to/features.jsonl \
    --num_samples 40000
```

**Step 2** -- Bin features and create splits:

```bash
python scripts/probing/split_probing_datasets.py \
    --input_path /path/to/features.jsonl \
    --output_path /path/to/unified.jsonl
```