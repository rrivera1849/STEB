# Clustering

Evaluates how well embeddings form clusters that align with style-based class labels. Episodes are embedded and K-Means clustering is applied. Quality is measured using [V-measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html).

## Metrics

- **V-measure**: harmonic mean of homogeneity and completeness (higher is better)

## CLI

```bash
steb clustering rrivera1849/LUAR-MUD --dataset corpus-of-diverse-styles -e 5
```

## Datasets

List available datasets:

```bash
steb clustering --list-datasets
```
