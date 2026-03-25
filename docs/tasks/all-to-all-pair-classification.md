# All-to-All Pair Classification

Evaluates whether embeddings can distinguish same-class vs. different-class text groups using cosine similarity across all pairs.

## Metrics

- **EER** (Equal Error Rate): lower is better
- **AUC**: higher is better
- **AUC@FPR**: AUC at false positive rate thresholds 0.01, 0.05, 0.10, 0.20, 0.30, 0.50

## CLI

```bash
steb all_to_all_pair_classification rrivera1849/LUAR-MUD --dataset corpus-of-diverse-styles -e 5
```

## Datasets

List available datasets:

```bash
steb all_to_all_pair_classification --list-datasets
```
