# Order Alignment

Evaluates how well embeddings preserve the ordering of graded stylistic dimensions (e.g., formality levels). Given two text sets ordered by style intensity, the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) finds the optimal alignment between positions. This generalizes the STEL task.

## Metrics

- **acc_mean**: baseline alignment accuracy
- **distractor_acc_mean**: accuracy with distractors injected

## Distractor Variants

The task includes distractor variants where items from one set are injected into another, testing robustness to style distractors.

### Variant 1: Replace Last (Least-Intense) Position

The last (least-intense) item from Set I is removed and injected into Set II, replacing its last position. The algorithm must align the reduced Set I to the corrupted Set II.

### Variant 2: Replace First (Most-Intense) Position

The first (most-intense) item from Set I is removed and injected into Set II, replacing its first position.

### Combined Metric

The `distractor_acc_mean` metric is the average accuracy across both distractor variants, testing whether the model's embeddings are robust to style distractors from either end of the intensity spectrum.

For full details on how the Hungarian algorithm is applied, see the [Hungarian Algorithm documentation](hungarian-algorithm.md).

## CLI

```bash
steb order_alignment rrivera1849/LUAR-MUD --dataset <dataset_name> -e 5
```

## Datasets

List available datasets:

```bash
steb order_alignment --list-datasets
```
