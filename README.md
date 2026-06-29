# STEB: Style Text Embedding Benchmark

STEB is a framework for evaluating style text embeddings across a variety of tasks and datasets. It is modular and extensible, making it straightforward to add new models, datasets, and evaluation tasks.

**[Leaderboard](https://rrivera1849.github.io/STEB/leaderboard/)** — current results for every benchmarked model under both the Operational and Definitional STEB scores. New submissions land here automatically when a contributor's PR merges; see [Submitting your model](#submitting-your-model) below.

## Installation

```bash
git clone https://github.com/rrivera1849/STEB.git
cd STEB
python3 -m venv venv
source venv/bin/activate
pip install -e .
./download_datasets.sh
```

The `download_datasets.sh` script invokes Python tools (`gdown`, etc.) declared in `requirements.txt`, so the venv must be active when running it. Use `--purge` to force a clean re-download.

To download to a non-default location, pass the target directory as an argument. You then need to tell STEB where to look, either via the `STEB_RAW_DATASETS_DIR` environment variable or via `config.ini` (see [Configuration](#configuration)):

```bash
./download_datasets.sh /path/to/raw_datasets
export STEB_RAW_DATASETS_DIR=/path/to/raw_datasets
```

## Quick Start

Run the standard STEB benchmark — the configuration reported in the paper:

```bash
steb "rrivera1849/LUAR-MUD"
```

Run a specific task on a specific dataset, for example, the PAN13 authorship verification benchmark:

```bash
steb pre_defined_pair_classification "rrivera1849/LUAR-MUD" \
    --dataset pan13_authorship_verification_english_test
```

## Configuration

STEB resolves four directory paths from environment variables, a `config.ini` file, and built-in defaults. Resolution order is:

1. Environment variable
2. `./config.ini` (in the current working directory)
3. `~/.steb/config.ini` (per-user fallback)
4. Built-in default

| Variable | `config.ini` key | Default | Description |
|---|---|---|---|
| `STEB_RAW_DATASETS_DIR` | `raw_datasets_dir` | `./raw_datasets` | Raw downloaded datasets |
| `STEB_RESULTS_DIR` | `results_dir` | `./results` | Evaluation results |
| `STEB_PROCESSED_DATA_DIR` | `processed_dataset_dir` | `~/.local/share/steb/processed_datasets` | Processed dataset cache |

### Environment variables

```bash
export STEB_RAW_DATASETS_DIR=/path/to/raw_datasets
export STEB_RESULTS_DIR=/path/to/results
export STEB_PROCESSED_DATA_DIR=/path/to/processed
```

### `config.ini`

Equivalent setup via `./config.ini` (project-local) or `~/.steb/config.ini` (user-wide):

```ini
[Application_Paths]
raw_datasets_dir = /path/to/raw_datasets
results_dir = /path/to/results
processed_dataset_dir = /path/to/processed_datasets
```

## Documentation

For the full reference — CLI flags, task descriptions, supported model types, dataset config schema, and guides for adding new models or datasets, see:

**[STEB Documentation](https://rrivera1849.github.io/STEB/)** *(also browsable as Markdown under [`docs/`](docs/))*

## Submitting your model

To get your model on the [public leaderboard](https://rrivera1849.github.io/STEB/leaderboard/), the short version is:

```bash
./scripts/download_results.sh                                   # 1. fetch canonical baselines (~10 MB)
STEB_RESULTS_DIR=./submitted_results steb "<org/your-model>"    # 2. run STEB into the community tree
python -m scripts.benchmark_clustering                          # 3. regenerate scores.xlsx, see your model
# 4. append a 4-key entry to SUBMISSIONS.yaml, add yourself to scripts/models_all.txt
python scripts/validate_submission.py                           # 5. validate, then open the PR
```

The `SUBMISSIONS.yaml` entry is:

```yaml
- short_name: <your-short-name>
  hf_id: <org/your-model>
  run_command: steb "<org/your-model>"
  contributor: <your-github-handle>
```

## Citation

If you use STEB in your research, please cite:

```bibtex
@article{TODO,
  title  = {STEB: A Style Text Embedding Benchmark},
  author = {Rivera Soto, Rafael A. and Wegmann, Anna and Aggazzotti, Cristina},
  year   = {2025}
}
```

## License

STEB is released under the [Apache License 2.0](LICENSE).
