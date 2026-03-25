# CLI Usage

STEB provides a command-line interface for running evaluations and managing datasets.

## Running Evaluations

### Run All Tasks

```bash
steb all "rrivera1849/LUAR-MUD" -e 1
```

### Run a Specific Task

```bash
steb clustering "rrivera1849/LUAR-MUD" --dataset "sms_spam" -e 1
```

### Run with a Preset

```bash
steb --preset fast "rrivera1849/LUAR-MUD"
```

### Common Options

| Option | Description |
|---|---|
| `-e`, `--episode-sizes` | Number of atomic units to form a writing sample |
| `--n-episodes-per-class` | Number of examples per class (default: 50) |
| `--batch-size` | Batch size for embedding (default: 32) |
| `--output-folder` | Folder to save results to |
| `--force-reload` | Force reload datasets |
| `--progress-bar` | Show a progress bar |
| `--seed` | Random seed (default: 42) |

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
