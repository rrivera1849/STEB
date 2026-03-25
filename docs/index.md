# STEB: Style Text Embedding Benchmark

STEB is a framework for evaluating style text embeddings across a variety of tasks and datasets. It is modular and extensible, making it straightforward to add new models, datasets, and evaluation tasks.

## Features

- **Multiple evaluation tasks**: clustering, pair classification, order alignment, retrieval, and probing
- **Flexible model support**: encoder models, causal LMs, and LUAR models with automatic detection
- **CLI and Python API**: run evaluations from the command line or programmatically
- **Extensible**: add new models, datasets, and tasks with minimal boilerplate
- **Preset configurations**: run standard evaluation suites with a single command

## Quick Start

```bash
# Install
git clone https://github.com/rrivera1849/STEB.git
cd STEB
pip install -e .

# Run an evaluation
steb clustering "rrivera1849/LUAR-MUD" --dataset corpus-of-diverse-styles -e 5
```

```python
import steb

model = steb.get_model("rrivera1849/LUAR-MUD")
datasets = steb.get_supported_datasets(task_name="clustering")
steb.evaluate(model, datasets=datasets, task_name="clustering", episode_sizes=[1])
```

## Next Steps

- [Getting Started](getting-started.md) — installation and configuration
- [CLI Usage](usage/cli.md) — command-line reference
- [Python API](usage/python-api.md) — programmatic usage
- [Tasks Overview](tasks/overview.md) — learn about the evaluation tasks
