# Retrieval

Evaluates how well embeddings retrieve style-matched texts. Given query and target sets, measures how well the correct target is ranked.

## Metrics

- **MRR** (Mean Reciprocal Rank)
- **Mean Rank**
- **Recall@K** for K = 1, 8, 16, 32, 64, 128

## CLI

```bash
steb retrieval rrivera1849/LUAR-MUD --dataset <dataset_name> -e 50 --n-episodes-per-class 1
```

For datasets with the standard JSONL format (`text`, `label`, `is_query` fields), the default retrieval loader in `steb/loaders/retrieval.py` is used. See `steb/steb_datasets/dummy_retrieval/config.json` for an example.

## Datasets

List available datasets:

```bash
steb retrieval --list-datasets
```
