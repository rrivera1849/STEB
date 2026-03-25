# Adding Datasets

## Quick Start

The fastest way to add a dataset:

```bash
# Scaffold the directory and config
steb new-dataset my_dataset --type huggingface

# Edit the generated config.json (set path, split, tasks, etc.)

# Validate your config
steb validate
```

This creates `steb/steb_datasets/my_dataset/config.json` (and a stub `loader.py` for custom datasets). The dataset is automatically discovered once the config exists.

## Config Schema

### HuggingFace Dataset

```json
{
  "dataset_name": "my_dataset",
  "type": "huggingface",
  "record_handler": {
    "text_getter": "text",
    "label_getter": "label"
  },
  "loader_kwargs": {
    "path": "huggingface/dataset-id",
    "split": "train"
  },
  "tasks": {
    "clustering": { "processor": "clustering" },
    "all_to_all_pair_classification": { "processor": "all_to_all_pair_classification" }
  }
}
```

### Custom Dataset

For custom datasets, replace `loader_kwargs` with `data_dir` and `loader_function`:

```json
{
  "dataset_name": "my_dataset",
  "type": "custom",
  "data_dir": "my_dataset",
  "loader_function": "load_my_dataset",
  "record_handler": {
    "text_getter": "text",
    "label_getter": "label"
  },
  "tasks": {
    "clustering": { "processor": "clustering" }
  }
}
```

## Loader Location Convention

- **Shared loaders** (used by multiple datasets, e.g. PAN, Fisher): `steb/loaders/`
- **Dataset-specific loaders**: `steb/steb_datasets/<name>/loader.py`

## Checklist

1. Run `steb new-dataset <name> --type huggingface` or `steb new-dataset <name> --type custom`
2. Fill in the `tasks` field in `config.json`
3. For custom datasets, implement the loader and add the download step to `download_datasets.sh`
4. Run `steb validate` to check the config
