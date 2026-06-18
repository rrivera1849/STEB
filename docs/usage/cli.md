# CLI Usage

STEB provides a command-line interface for running evaluations and managing datasets.

## Running Evaluations

### Run the Standard Benchmark

```bash
steb "rrivera1849/LUAR-MUD"
```

This is the configuration reported in the paper. With no task or `--preset`, `steb <MODEL>` is equivalent to `steb <MODEL> --preset benchmark`.

### Run with a Preset

```bash
steb --preset fast "rrivera1849/LUAR-MUD"
steb --preset sanity "rrivera1849/LUAR-MUD"
```

### Run a Specific Task

```bash
steb clustering "rrivera1849/LUAR-MUD" --dataset "sms_spam" -e 1
```

### Run All Tasks

```bash
steb all "rrivera1849/LUAR-MUD"
```

Uses per-task defaults from `TASK_DEFAULTS`.

### Common Options

| Option | Description |
|---|---|
| `-e`, `--episode-sizes` | Number of atomic units to form a writing sample |
| `--n-episodes-per-class` | Number of examples per class. Pass an integer or `auto`. Default: per-task default from `TASK_DEFAULTS` |
| `--batch-size` | Batch size for embedding (default: 32) |
| `--output-folder` | Folder to save results to |
| `--force-reload` | Force reload datasets |
| `--force-rerun` | Re-run all tasks even if metrics already exist |
| `--force-rerun-oa` | Re-run the `order_alignment` task only |
| `--progress-bar` | Show a progress bar |
| `--seed` | Random seed (default: 42) |
| `--truncate` | Truncate each text to the token cap instead of chunking and mean-pooling |
| `--max-tokens` | Per-text token cap. Capped at the model's native max. Default: model max |

### List Available Datasets

```bash
steb clustering --list-datasets
steb retrieval --list-datasets
```

## Utility Commands

### Validate Configs

Check that all dataset `config.json` files are well-formed:

```bash
steb validate
```

### Scaffold a New Dataset

```bash
# HuggingFace dataset
steb new-dataset my_dataset --type huggingface

# Custom dataset with a loader stub
steb new-dataset my_dataset --type custom
```

See [Adding Datasets](../developer/adding-datasets.md) for the full workflow.

### Preview Dataset Statistics

Inspect class counts, resolved `n_episodes_per_class`, and any dropped classes for an upcoming benchmark run, without loading a model:

```bash
# Preview the full benchmark preset
steb preview --preset benchmark

# Preview a single task across all supported datasets
steb preview --task clustering

# Preview a single dataset
steb preview --dataset corpus-of-diverse-styles
```

By default the report is written to `preview.txt`; override with `-o`.
