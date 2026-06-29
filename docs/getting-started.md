# Getting Started

## Installation

```bash
git clone https://github.com/rrivera1849/STEB.git
cd STEB
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Downloading Datasets

Some datasets need to be downloaded before use:

```bash
./download_datasets.sh
```

The script skips datasets that have already been downloaded. Use `--purge` to force a clean re-download.

## Configuration

By default, STEB looks for raw datasets in `./raw_datasets` relative to the working directory. All paths are configurable via environment variables or a `config.ini` file.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STEB_RESULTS_DIR` | `./results` | Where evaluation results are saved |
| `STEB_PROCESSED_DATA_DIR` | `~/.local/share/steb/processed_datasets` | Processed dataset cache |
| `STEB_RAW_DATASETS_DIR` | `./raw_datasets` | Raw downloaded datasets |

The HuggingFace `load_dataset` cache (used for HF-format datasets in the suite) is managed by HuggingFace itself; set `HF_DATASETS_CACHE` or `HF_HOME` to relocate it.

### Config File

These can also be set in a `config.ini` file (in the current directory or `~/.steb/config.ini`):

```ini
[Application_Paths]
processed_dataset_dir = /path/to/your/processed_datasets
results_dir = /path/to/your/results
raw_datasets_dir = /path/to/your/raw_datasets
```
