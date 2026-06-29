# Leaderboard

The canonical STEB leaderboard. Numbers reproduce the headline tables in the [STEB paper](https://github.com/rrivera1849/STEB/blob/main/STEB_paper.pdf) and are regenerated whenever the maintainer refreshes `scores.xlsx` from the latest benchmark runs.

There are two STEB scores, with different aggregation philosophies:

- **Operational** mirrors how the field has historically organized style work: macro-average within auto-discovered redundancy clusters per task, then across tasks. See `STEB_operational` below.
- **Definitional** scores embeddings against the style definition of Wegmann et al. (2026): the average of three axes — Object of Study (Genre, Register, Time, Demographics, Dialect, Idiolect), Linguistic Features, and Content Independence. See `STEB_definitional` below.

To submit a new model, see the [Submitting your model](https://github.com/rrivera1849/STEB/blob/main/README.md#submitting-your-model) section of the README.

## STEB (Operational)

Sorted by `STEB_score (avg)` descending. **Bold** = best per column, *italic* = second best.

| Model | clustering (v_measure) | all_to_all_pair_classification (auc) | pre_defined_pair_classification (auc) | order_alignment (distractor_acc_mean) | retrieval (mrr) | probing (average) | STEB_score (avg) |
|---|---|---|---|---|---|---|---|
| LUAR-CRUD | 19.65 | 61.00 | *73.12* | 19.80 | **77.64** | 53.75 | **50.82** |
| star | **25.92** | **64.56** | **73.28** | 23.93 | 66.96 | 49.86 | *50.75* |
| LUAR-MUD | 20.03 | 62.33 | 72.33 | 20.81 | *75.67* | 53.04 | 50.70 |
| styledistance | 18.46 | 59.82 | 66.21 | **44.02** | 49.61 | 57.87 | 49.33 |
| deberta-v3-large | *21.44* | 59.37 | 70.45 | 24.23 | 63.24 | 53.41 | 48.69 |
| multilingual-style-representation | 16.67 | *64.34* | 71.01 | 21.85 | 63.49 | 51.01 | 48.06 |
| roberta-large | 13.21 | 60.97 | 68.79 | 20.71 | 61.07 | *57.92* | 47.11 |
| bert-large-cased | 12.02 | 60.99 | 67.11 | 21.57 | 58.88 | 55.16 | 45.96 |
| deberta-v3-base | 19.42 | 58.35 | 66.82 | 23.41 | 55.17 | 51.73 | 45.81 |
| roberta-base | 12.59 | 60.97 | 66.49 | 22.47 | 50.63 | **61.32** | 45.74 |
| Style-Embedding | 15.28 | 59.71 | 66.43 | 33.81 | 45.87 | 52.54 | 45.61 |
| bert-large-uncased | 11.18 | 60.44 | 66.92 | 21.18 | 60.23 | 52.48 | 45.41 |
| ModernBERT-large | 16.58 | 60.19 | 65.38 | 20.56 | 52.17 | 54.06 | 44.82 |
| ModernBERT-base | 14.05 | 59.28 | 64.13 | 19.20 | 51.25 | 53.91 | 43.63 |
| styledistance_synthetic_only | 13.46 | 57.02 | 57.67 | *42.76* | 31.53 | 56.69 | 43.19 |
| e5-large-v2 | 11.06 | 60.20 | 64.70 | 15.82 | 49.91 | 49.03 | 41.79 |
| e5-base-v2 | 9.28 | 60.40 | 63.42 | 15.26 | 49.62 | 52.13 | 41.69 |
| gte-large-en-v1.5 | 11.48 | 60.99 | 62.39 | 16.20 | 49.60 | 48.87 | 41.59 |
| tfidfngrams_fineweb_sample10bt_1-2grams.pkl | 6.78 | 57.18 | 64.45 | 13.72 | 50.95 | 55.74 | 41.47 |
| tfidfngrams_mud_subset_1-2grams.pkl | 6.79 | 56.93 | 64.45 | 13.84 | 49.77 | 55.54 | 41.22 |
| bge-base-en-v1.5 | 7.93 | 59.40 | 62.97 | 15.82 | 51.44 | 49.45 | 41.17 |
| tfidfngrams_mud_subset_1-3grams.pkl | 6.52 | 56.68 | 64.25 | 13.84 | 50.26 | 55.34 | 41.15 |
| gte-base-en-v1.5 | 10.76 | 60.04 | 60.33 | 15.62 | 49.36 | 50.64 | 41.13 |
| tfidfngrams_fineweb_sample10bt_1-3grams.pkl | 6.37 | 56.97 | 64.45 | 13.67 | 49.00 | 55.51 | 40.99 |
| surface_pos.yaml | 9.67 | 56.88 | 60.48 | 25.87 | 39.52 | 53.30 | 40.95 |
| jina-embeddings-v3 | 7.90 | 60.58 | 64.60 | 17.11 | 47.35 | 45.62 | 40.53 |
| bge-large-en-v1.5 | 7.15 | 59.84 | 63.81 | 16.23 | 48.79 | 47.23 | 40.51 |
| all-mpnet-base-v2 | 8.48 | 60.41 | 63.87 | 14.62 | 48.99 | 45.64 | 40.34 |
| lisa_checkpoint | 12.74 | 60.91 | 62.40 | 16.48 | 42.96 | 46.38 | 40.31 |
| opt-1.3b | 3.83 | 52.40 | 55.59 | 24.08 | 39.07 | 45.82 | 36.80 |
| Qwen3.5-4B-Base | 4.23 | 52.80 | 57.41 | 24.88 | 37.15 | 43.02 | 36.58 |
| functionwordfreq | 7.17 | 55.17 | 60.88 | 14.51 | 43.14 | 37.98 | 36.47 |
| Qwen3.5-2B-Base | 3.97 | 52.46 | 57.09 | 24.61 | 35.13 | 42.61 | 35.98 |
| Qwen3.5-0.8B-Base | 4.21 | 52.73 | 57.23 | 25.07 | 32.83 | 43.30 | 35.89 |
| gpt2-xl | 4.42 | 51.68 | 53.49 | 26.32 | 28.82 | 48.66 | 35.57 |
| mstyledistance | 7.35 | 53.83 | 53.71 | 31.38 | 20.04 | 45.96 | 35.38 |
| Qwen3-Embedding-8B | 2.67 | 51.61 | 52.79 | 23.11 | 35.17 | 42.96 | 34.72 |
| Qwen2-0.5B | 3.77 | 51.59 | 52.78 | 25.51 | 25.70 | 42.34 | 33.62 |
| Qwen3-0.6B-Base | 3.73 | 51.43 | 53.14 | 26.01 | 23.67 | 41.78 | 33.29 |
| neurobiber | 8.24 | 55.10 | 57.27 | 14.13 | 9.06 | 51.64 | 32.57 |

## STEB (Definitional)

Sorted by `Definitional score` descending. **Bold** = best per column, *italic* = second best.

| Model | Genre | Register | Time | Demographics | Dialect | Idiolect | Object of Study | Linguistic Features | Content Independence | Definitional score |
|---|---|---|---|---|---|---|---|---|---|---|
| styledistance | 62.84 | 40.01 | 46.21 | 33.67 | **56.91** | 60.03 | 49.94 | 74.02 | *44.23* | **56.06** |
| styledistance_synthetic_only | 55.45 | 38.81 | 43.27 | 33.13 | *55.64* | 45.26 | 45.26 | 70.87 | **46.10** | *54.08* |
| Style-Embedding | 62.43 | 36.09 | 41.52 | 33.03 | 54.76 | 58.72 | 47.76 | 65.85 | 32.31 | 48.64 |
| deberta-v3-large | 63.26 | 38.25 | *46.94* | 35.37 | 43.37 | 68.26 | 49.24 | 69.56 | 18.72 | 45.84 |
| roberta-base | 63.79 | **42.52** | 45.94 | 32.84 | 40.16 | 59.83 | 47.51 | **77.95** | 10.27 | 45.25 |
| star | **68.68** | 39.79 | **47.25** | **40.99** | 49.39 | 72.24 | **53.06** | 67.66 | 14.08 | 44.93 |
| roberta-large | 65.84 | 40.01 | 46.43 | 33.07 | 42.07 | 66.21 | 48.94 | *75.48* | 9.86 | 44.76 |
| deberta-v3-base | 58.34 | 36.63 | 44.97 | 33.72 | 46.08 | 62.29 | 47.01 | 68.28 | 18.49 | 44.59 |
| surface_pos.yaml | 54.60 | 38.59 | 34.43 | 32.44 | 35.49 | 51.19 | 41.12 | 67.36 | 24.89 | 44.46 |
| mstyledistance | 49.41 | 32.83 | 35.04 | 32.25 | 42.83 | 37.23 | 38.27 | 56.17 | 38.32 | 44.25 |
| LUAR-MUD | *66.67* | 38.72 | 42.14 | 35.53 | 44.30 | *76.11* | *50.58* | 70.50 | 11.29 | 44.12 |
| bert-large-cased | 66.24 | *41.99* | 45.11 | 34.79 | 41.23 | 64.12 | 48.91 | 73.58 | 9.22 | 43.90 |
| ModernBERT-large | 63.99 | 39.77 | 44.29 | 32.75 | 42.28 | 60.13 | 47.20 | 73.38 | 10.35 | 43.64 |
| LUAR-CRUD | 65.80 | 38.14 | 41.30 | 35.32 | 41.52 | **76.99** | 49.85 | 70.37 | 10.57 | 43.60 |
| ModernBERT-base | 62.23 | 38.76 | 43.14 | 32.67 | 37.71 | 58.98 | 45.58 | 72.90 | 9.64 | 42.71 |
| multilingual-style-representation | 66.39 | 39.08 | 41.98 | *37.05* | 41.52 | 68.69 | 49.12 | 66.52 | 12.35 | 42.66 |
| bert-large-uncased | 62.24 | 39.90 | 45.27 | 32.80 | 44.03 | 64.81 | 48.17 | 68.55 | 10.89 | 42.54 |
| Qwen3.5-0.8B-Base | 46.66 | 35.59 | 40.51 | 28.29 | 32.34 | 45.08 | 38.08 | 60.61 | 26.03 | 41.57 |
| Qwen2-0.5B | 44.76 | 36.01 | 40.57 | 28.24 | 34.66 | 39.32 | 37.26 | 60.46 | 26.76 | 41.49 |
| gpt2-xl | 41.36 | 32.22 | 36.15 | 27.29 | 34.00 | 40.83 | 35.31 | 58.54 | 29.14 | 41.00 |
| Qwen3.5-2B-Base | 45.71 | 34.81 | 39.04 | 28.33 | 33.39 | 46.08 | 37.89 | 59.49 | 25.49 | 40.96 |
| Qwen3.5-4B-Base | 44.42 | 34.62 | 38.93 | 28.43 | 34.70 | 47.35 | 38.07 | 58.63 | 25.72 | 40.81 |
| Qwen3-0.6B-Base | 42.81 | 34.26 | 38.22 | 28.26 | 35.77 | 38.48 | 36.30 | 57.99 | 28.12 | 40.80 |
| opt-1.3b | 43.20 | 33.19 | 38.79 | 28.76 | 30.68 | 46.92 | 36.93 | 57.85 | 24.29 | 39.69 |
| Qwen3-Embedding-8B | 39.57 | 33.88 | 37.26 | 26.92 | 27.05 | 43.96 | 34.77 | 57.68 | 24.10 | 38.85 |
| e5-base-v2 | 55.91 | 37.26 | 39.72 | 30.50 | 22.85 | 57.80 | 40.67 | 69.27 | 5.11 | 38.35 |
| e5-large-v2 | 57.58 | 37.70 | 40.81 | 31.38 | 22.16 | 58.53 | 41.36 | 67.54 | 5.22 | 38.04 |
| functionwordfreq | 51.00 | 35.58 | 43.56 | 31.56 | 27.06 | 53.12 | 40.31 | 63.21 | 9.52 | 37.68 |
| lisa_checkpoint | 56.89 | 37.95 | 37.38 | 32.95 | 27.19 | 54.00 | 41.06 | 64.02 | 7.10 | 37.40 |
| tfidfngrams_mud_subset_1-2grams.pkl | 49.91 | 33.96 | 36.19 | 31.67 | 27.71 | 57.98 | 39.57 | 66.90 | 5.66 | 37.38 |
| tfidfngrams_fineweb_sample10bt_1-2grams.pkl | 49.19 | 33.79 | 36.20 | 31.70 | 26.31 | 58.70 | 39.31 | 67.13 | 5.55 | 37.33 |
| gte-base-en-v1.5 | 53.62 | 36.51 | 39.44 | 31.76 | 23.10 | 56.56 | 40.16 | 65.77 | 5.31 | 37.08 |
| tfidfngrams_mud_subset_1-3grams.pkl | 48.90 | 33.75 | 36.37 | 31.84 | 25.33 | 58.05 | 39.04 | 66.55 | 5.62 | 37.07 |
| bge-base-en-v1.5 | 54.69 | 36.67 | 36.96 | 30.75 | 23.10 | 58.25 | 40.07 | 65.79 | 5.22 | 37.03 |
| bge-large-en-v1.5 | 56.15 | 37.31 | 38.23 | 29.71 | 22.47 | 57.35 | 40.20 | 65.54 | 5.28 | 37.01 |
| tfidfngrams_fineweb_sample10bt_1-3grams.pkl | 48.49 | 33.55 | 36.30 | 31.75 | 24.81 | 57.64 | 38.76 | 66.68 | 5.47 | 36.97 |
| jina-embeddings-v3 | 56.24 | 37.51 | 39.77 | 31.10 | 23.74 | 57.25 | 40.94 | 63.80 | 5.53 | 36.75 |
| gte-large-en-v1.5 | 57.09 | 36.67 | 41.20 | 31.38 | 24.01 | 57.26 | 41.27 | 63.40 | 5.45 | 36.71 |
| all-mpnet-base-v2 | 56.07 | 37.55 | 39.03 | 31.47 | 23.31 | 57.50 | 40.82 | 62.18 | 5.06 | 36.02 |
| neurobiber | 47.31 | 32.38 | 34.49 | 31.46 | 28.07 | 34.04 | 34.62 | 64.11 | 7.99 | 35.57 |
