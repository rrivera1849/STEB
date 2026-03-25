# Pre-defined Pair Classification

Same as all-to-all pair classification, but operates on datasets with pre-defined pairs (e.g., authorship verification). Episode size and episodes-per-class are set automatically.

## Metrics

- **EER** (Equal Error Rate): lower is better
- **AUC**: higher is better
- **AUC@FPR**: AUC at false positive rate thresholds 0.01, 0.05, 0.10, 0.20, 0.30, 0.50

## CLI

```bash
steb pre_defined_pair_classification rrivera1849/LUAR-MUD --dataset pan15_authorship_verification_english_test
```

## Datasets

List available datasets:

```bash
steb pre_defined_pair_classification --list-datasets
```
