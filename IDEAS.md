# Ideas for Improvement Before Adding More Datasets

## Config Validation

Right now there is no validation of `config.json` files. A malformed config silently
breaks at runtime. We should add a validation step that checks:

- Required fields are present (`type`, `record_handler`, `tasks`)
- `type` is one of `huggingface` or `custom`
- HuggingFace configs have `loader_kwargs` with at least `path` and `split`
- Custom configs have `loader_function` and `data_dir`
- Every task listed in `tasks` maps to an existing processor
- `record_handler` has `text_getter` and `label_getter` (or a `custom_record_handler_function`)

This could be a JSON Schema file or a simple Python validation function that runs on
import or via a CLI command like `steb validate`.

## Dataset Template / Scaffold Command

Adding a new dataset requires creating a directory, writing a config.json, and
optionally writing a loader. A CLI command like `steb new-dataset <name> --type huggingface`
that scaffolds the directory with a template config.json would reduce errors and
speed up the process.

## CI / Automated Testing

There are no GitHub Actions workflows. Before the dataset count grows, we should add:

- A workflow that runs `pytest` on every PR (at minimum the fast, non-model tests)
- A config validation check that ensures all `config.json` files are well-formed
- A lint step (ruff or similar) to enforce import ordering and style

## Standardize Loader Location

Custom loaders live in two places: `steb/loaders/` (reusable) and
`steb/steb_datasets/<name>/loader.py` (dataset-specific). The convention isn't
documented anywhere. We should decide on a rule, e.g.:

- If a loader is shared by multiple datasets (PAN, Fisher) -> `steb/loaders/`
- If a loader is specific to one dataset -> `steb/steb_datasets/<name>/loader.py`

And document it so new contributors follow the same pattern.

## Dataset Size / Quality Checks

When a dataset is loaded, we only discover problems (e.g., too few samples per class)
deep inside `get_valid_labels()`. A pre-flight check after loading that logs:

- Number of classes and samples per class
- Whether episode_size * n_episodes_per_class is achievable for each class
- Warnings for classes that will be dropped

would make debugging much faster, especially for new datasets where the data shape
is unfamiliar.

## Reproducibility of Processed Datasets

Processed datasets are cached to `~/.local/share/steb/processed_datasets/` keyed by
`{dataset_name}_{n_episodes_per_class}_{episode_size}.json`. The cache key does NOT
include the random seed, so changing the seed without `--force-reload` silently uses
stale cached data. Either include the seed in the cache key, or document this clearly.

NOTE: Include the key in the cache key.

## Integration Test Coverage

`test.sh` only tests one task on one dataset. As the number of datasets grows, we
should have at least one integration test per task type (clustering, retrieval,
pair classification, order alignment, probing) that runs end-to-end on a small/dummy
dataset. The dummy_order_alignment and dummy_retrieval datasets are a good start;
we should add dummy datasets for the remaining tasks too.

NOTE: Please do so, and add it to tests/
