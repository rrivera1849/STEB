# Probing

Trains a logistic regression probe on frozen embeddings to evaluate what linguistic properties are encoded. Uses train/val/test splits defined per-sample in the dataset.

## Metrics

- **Per-task accuracy**: accuracy on each individual probing task
- **Average accuracy**: mean accuracy across all probing tasks

## CLI

```bash
steb probing rrivera1849/LUAR-MUD --dataset <dataset_name>
```

Episode size and episodes-per-class are set automatically for probing tasks.

## Datasets

List available datasets:

```bash
steb probing --list-datasets
```
