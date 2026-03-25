# Mapping Sentence Sets to the Hungarian Algorithm

## Problem Setup

We have two sets of sentences:

| Set I (fixed) | Set II (can vary) |
|:-:|:-:|
| S1 | T1 |
| S2 | T2 |
| S3 | T3 |

**Set I** (S1, S2, S3) is held **fixed**, while the ordering of **Set II** (T1, T2, T3) **can vary**. The goal is to find the optimal one-to-one assignment of sentences in Set II to sentences in Set I.

## Bipartite Matching

This is a bipartite matching problem: each sentence T*i* from Set II must be assigned to exactly one sentence S*j* from Set I, with no overlaps.

```
Set I        Set II
 S1 ───────── T1
 S2 ───────── T2
 S3 ───────── T3
```

## Cost Matrix

To apply the Hungarian algorithm, we construct a **cost matrix** where each entry `d(Ti, Sj)` represents the distance (or dissimilarity) between sentence T*i* and sentence S*j*:

|            | S1         | S2         | S3         |
|:----------:|:----------:|:----------:|:----------:|
| **T1**     | d(T1, S1)  | d(T1, S2)  | d(T1, S3)  |
| **T2**     | d(T2, S1)  | d(T2, S2)  | d(T2, S3)  |
| **T3**     | d(T3, S1)  | d(T3, S2)  | d(T3, S3)  |

Here, the rows correspond to **"workers"** (T1–T3) and the columns correspond to **"tasks"** (S1–S3) in the classic Hungarian algorithm terminology.

## Objective

The Hungarian algorithm finds the assignment that **minimizes** the total distance:

$$\min \sum_{i} d(X_i, S_i)$$

where $X_i$ is the sentence from Set II assigned to $S_i$.

In other words, we seek a permutation of T1, T2, T3 such that the sum of pairwise distances to S1, S2, S3 is minimized.

## Distractor Variants

In the order alignment task, positions are ordered by style intensity (most intense to least intense). Beyond the baseline alignment described above, two **distractor variants** test whether the model can still align correctly when one position in Set II is replaced by a "distractor" — an item taken from Set I that does not belong in Set II.

### Distractor Variant 1: Replace Last (Least-Intense) Position

The last (least-intense) item from Set I is removed from Set I and injected into Set II, replacing its last position. The algorithm must then align the *reduced* Set I to the *corrupted* Set II.

**Before (baseline):**

| Set I (reference) | Set II (to align) |
|:-:|:-:|
| S1 (most intense) | T1 (most intense) |
| S2 | T2 |
| S3 (least intense) | T3 (least intense) |

**After distractor injection:**

| Set I (reference) | Set II (to align) |
|:-:|:-:|
| S1 (most intense) | T1 (most intense) |
| S2 | T2 |
| ~~S3~~ *(removed)* | ~~T3~~ → **S3** *(distractor)* |

Set I shrinks to (S1, S2), while Set II becomes (T1, T2, S3). The cost matrix is now rectangular:

|            | T1         | T2         | S3 (distractor) |
|:----------:|:----------:|:----------:|:----------:|
| **S1**     | d(S1, T1)  | d(S1, T2)  | d(S1, S3)  |
| **S2**     | d(S2, T1)  | d(S2, T2)  | d(S2, S3)  |

The Hungarian algorithm finds the optimal assignment. A correct alignment maps S1 → T1 and S2 → T2, leaving the distractor S3 unmatched. Accuracy is measured over the positions in the reduced Set I (expected positions 0 and 1).

### Distractor Variant 2: Replace First (Most-Intense) Position

The first (most-intense) item from Set I is removed from Set I and injected into Set II, replacing its first position.

**After distractor injection:**

| Set I (reference) | Set II (to align) |
|:-:|:-:|
| ~~S1~~ *(removed)* | ~~T1~~ → **S1** *(distractor)* |
| S2 | T2 |
| S3 (least intense) | T3 (least intense) |

Set I shrinks to (S2, S3), while Set II becomes (S1, T2, T3). The cost matrix is:

|            | S1 (distractor) | T2         | T3         |
|:----------:|:----------:|:----------:|:----------:|
| **S2**     | d(S2, S1)  | d(S2, T2)  | d(S2, T3)  |
| **S3**     | d(S3, S1)  | d(S3, T2)  | d(S3, T3)  |

A correct alignment maps S2 → T2 and S3 → T3, leaving the distractor S1 unmatched. Since the reduced Set I starts at the second position, the expected positions are offset by 1 (expected positions 1 and 2).

### Combined Distractor Metric

The final reported metric, `distractor_acc_mean`, is the average accuracy across **both** distractor variants. This tests whether the model's embeddings are robust to style distractors from either end of the intensity spectrum — i.e., whether the model can avoid being "fooled" by an item that is stylistically very different but comes from a related source.
