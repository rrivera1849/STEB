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
| LUSR | *20.55* | 62.50 | 69.03 | 33.87 | 65.37 | *58.48* | **51.63** |
| star | **25.77** | **65.21** | **73.68** | 23.93 | 66.96 | 49.86 | *50.90* |
| LUAR-CRUD | 20.27 | 61.54 | *72.09* | 19.80 | **77.64** | 53.75 | 50.85 |
| LUAR-MUD | 20.40 | 62.87 | 71.24 | 20.81 | *75.67* | 53.04 | 50.67 |
| styledistance | 16.69 | 60.74 | 66.52 | **44.02** | 49.61 | 57.87 | 49.24 |
| deberta-v3-large | 20.09 | 59.89 | 71.32 | 24.23 | 63.24 | 53.41 | 48.70 |
| multilingual-style-representation | 19.85 | *64.59* | 70.72 | 21.85 | 63.49 | 51.01 | 48.59 |
| roberta-large | 15.93 | 61.50 | 69.15 | 20.71 | 61.07 | 57.92 | 47.71 |
| bert-large-cased | 14.67 | 61.48 | 67.33 | 21.57 | 58.88 | 55.16 | 46.51 |
| roberta-base | 15.03 | 61.44 | 66.91 | 22.47 | 50.63 | **61.32** | 46.30 |
| bert-large-uncased | 12.62 | 60.84 | 67.36 | 21.18 | 60.23 | 52.48 | 45.79 |
| deberta-v3-base | 16.54 | 58.73 | 67.64 | 23.41 | 55.17 | 51.73 | 45.54 |
| Style-Embedding | 13.59 | 60.61 | 66.36 | 33.81 | 45.87 | 52.54 | 45.46 |
| ModernBERT-large | 17.61 | 60.73 | 65.34 | 20.56 | 52.17 | 54.06 | 45.08 |
| ModernBERT-base | 15.30 | 59.67 | 64.32 | 19.20 | 51.25 | 53.91 | 43.94 |
| styledistance_synthetic_only | 12.45 | 57.73 | 58.36 | *42.76* | 31.53 | 56.69 | 43.25 |
| e5-large-v2 | 14.19 | 60.36 | 65.08 | 15.82 | 49.91 | 49.03 | 42.40 |
| gte-large-en-v1.5 | 14.62 | 61.08 | 63.34 | 16.20 | 49.60 | 48.87 | 42.29 |
| e5-base-v2 | 12.41 | 60.55 | 63.49 | 15.26 | 49.62 | 52.13 | 42.24 |
| gte-base-en-v1.5 | 13.70 | 60.32 | 60.78 | 15.62 | 49.36 | 50.64 | 41.74 |
| bge-base-en-v1.5 | 10.55 | 59.53 | 63.48 | 15.82 | 51.44 | 49.45 | 41.71 |
| tfidfngrams_fineweb_sample10bt_1-2grams.pkl | 7.05 | 57.39 | 65.24 | 13.72 | 50.95 | 55.74 | 41.68 |
| tfidfngrams_mud_subset_1-2grams.pkl | 7.01 | 57.09 | 65.23 | 13.84 | 49.77 | 55.54 | 41.41 |
| surface_pos.yaml | 11.06 | 57.50 | 60.94 | 25.87 | 39.52 | 53.30 | 41.36 |
| tfidfngrams_mud_subset_1-3grams.pkl | 6.74 | 56.82 | 65.05 | 13.84 | 50.26 | 55.34 | 41.34 |
| tfidfngrams_fineweb_sample10bt_1-3grams.pkl | 6.63 | 57.17 | 65.26 | 13.67 | 49.00 | 55.51 | 41.21 |
| jina-embeddings-v3 | 10.81 | 60.64 | 65.17 | 17.11 | 47.35 | 45.62 | 41.12 |
| bge-large-en-v1.5 | 9.78 | 59.98 | 64.40 | 16.23 | 48.79 | 47.23 | 41.07 |
| lisa_checkpoint | 14.75 | 61.12 | 61.90 | 16.48 | 42.96 | 46.38 | 40.60 |
| all-mpnet-base-v2 | 9.16 | 60.53 | 64.31 | 14.62 | 48.99 | 45.64 | 40.54 |
| opt-1.3b | 4.09 | 52.59 | 55.87 | 24.08 | 39.07 | 45.82 | 36.92 |
| Qwen3.5-4B-Base | 4.56 | 52.94 | 57.71 | 24.88 | 37.15 | 43.02 | 36.71 |
| functionwordfreq | 7.16 | 55.43 | 61.89 | 14.51 | 43.14 | 37.98 | 36.68 |
| Qwen3.5-2B-Base | 3.86 | 52.62 | 57.14 | 24.61 | 35.13 | 42.61 | 36.00 |
| Qwen3.5-0.8B-Base | 4.16 | 52.88 | 57.42 | 25.07 | 32.83 | 43.30 | 35.94 |
| gpt2-xl | 4.50 | 51.82 | 53.69 | 26.32 | 28.82 | 48.66 | 35.63 |
| mstyledistance | 7.43 | 54.18 | 53.97 | 31.38 | 20.04 | 45.96 | 35.49 |
| Qwen3-Embedding-8B | 3.11 | 51.69 | 52.52 | 23.11 | 35.17 | 42.96 | 34.76 |
| Qwen2-0.5B | 3.68 | 51.71 | 52.33 | 25.51 | 25.70 | 42.34 | 33.54 |
| Qwen3-0.6B-Base | 3.64 | 51.53 | 53.15 | 26.01 | 23.67 | 41.78 | 33.30 |
| neurobiber | 9.26 | 55.31 | 57.00 | 14.13 | 9.06 | 51.64 | 32.73 |

## STEB (Definitional)

Sorted by `Definitional score` descending. **Bold** = best per column, *italic* = second best.

| Model | Genre | Register | Time | Demographics | Dialect | Idiolect | Object of Study | Linguistic Features | Content Independence | Definitional score |
|---|---|---|---|---|---|---|---|---|---|---|
| styledistance | 62.84 | 40.01 | 46.21 | 33.67 | *63.04* | 60.03 | 50.97 | 74.02 | *44.23* | **56.40** |
| styledistance_synthetic_only | 55.45 | 38.81 | 43.27 | 33.13 | 62.63 | 45.26 | 46.43 | 70.87 | **46.10** | *54.47* |
| LUSR | *67.91* | 40.75 | **49.37** | 34.49 | **64.71** | 69.67 | *54.48* | 73.51 | 27.87 | 51.95 |
| Style-Embedding | 62.43 | 36.09 | 41.52 | 33.03 | 59.53 | 58.72 | 48.55 | 65.85 | 32.31 | 48.90 |
| deberta-v3-large | 63.26 | 38.25 | 46.94 | 35.37 | 44.70 | 68.26 | 49.46 | 69.56 | 18.72 | 45.91 |
| star | **68.68** | 39.79 | *47.25* | **40.99** | 58.22 | 72.24 | **54.53** | 67.66 | 14.08 | 45.42 |
| roberta-base | 63.79 | **42.52** | 45.94 | 32.84 | 42.95 | 59.83 | 47.98 | **77.95** | 10.27 | 45.40 |
| roberta-large | 65.84 | 40.01 | 46.43 | 33.07 | 48.90 | 66.21 | 50.08 | *75.48* | 9.86 | 45.14 |
| deberta-v3-base | 58.34 | 36.63 | 44.97 | 33.72 | 47.44 | 62.29 | 47.23 | 68.28 | 18.49 | 44.67 |
| LUAR-MUD | 66.67 | 38.72 | 42.14 | 35.53 | 50.42 | *76.11* | 51.60 | 70.50 | 11.29 | 44.46 |
| surface_pos.yaml | 54.60 | 38.59 | 34.43 | 32.44 | 31.15 | 51.19 | 40.40 | 67.36 | 24.89 | 44.22 |
| bert-large-cased | 66.24 | *41.99* | 45.11 | 34.79 | 46.74 | 64.12 | 49.83 | 73.58 | 9.22 | 44.21 |
| mstyledistance | 49.41 | 32.83 | 35.04 | 32.25 | 38.99 | 37.23 | 37.63 | 56.17 | 38.32 | 44.04 |
| ModernBERT-large | 63.99 | 39.77 | 44.29 | 32.75 | 47.52 | 60.13 | 48.08 | 73.38 | 10.35 | 43.94 |
| LUAR-CRUD | 65.80 | 38.14 | 41.30 | 35.32 | 45.39 | **76.99** | 50.49 | 70.37 | 10.57 | 43.81 |
| bert-large-uncased | 62.24 | 39.90 | 45.27 | 32.80 | 51.82 | 64.81 | 49.47 | 68.55 | 10.89 | 42.97 |
| multilingual-style-representation | 66.39 | 39.08 | 41.98 | *37.05* | 45.90 | 68.69 | 49.85 | 66.52 | 12.35 | 42.90 |
| ModernBERT-base | 62.23 | 38.76 | 43.14 | 32.67 | 40.03 | 58.98 | 45.97 | 72.90 | 9.64 | 42.84 |
| Qwen3.5-0.8B-Base | 46.66 | 35.59 | 40.51 | 28.29 | 29.13 | 45.08 | 37.54 | 60.61 | 26.03 | 41.40 |
| Qwen2-0.5B | 44.76 | 36.01 | 40.57 | 28.24 | 31.12 | 39.32 | 36.67 | 60.46 | 26.76 | 41.30 |
| gpt2-xl | 41.36 | 32.22 | 36.15 | 27.29 | 30.80 | 40.83 | 34.78 | 58.54 | 29.14 | 40.82 |
| Qwen3.5-2B-Base | 45.71 | 34.81 | 39.04 | 28.33 | 30.59 | 46.08 | 37.43 | 59.49 | 25.49 | 40.80 |
| Qwen3.5-4B-Base | 44.42 | 34.62 | 38.93 | 28.43 | 32.53 | 47.35 | 37.71 | 58.63 | 25.72 | 40.69 |
| Qwen3-0.6B-Base | 42.81 | 34.26 | 38.22 | 28.26 | 32.94 | 38.48 | 35.83 | 57.99 | 28.12 | 40.65 |
| opt-1.3b | 43.20 | 33.19 | 38.79 | 28.76 | 28.04 | 46.92 | 36.49 | 57.85 | 24.29 | 39.54 |
| Qwen3-Embedding-8B | 39.57 | 33.88 | 37.26 | 26.92 | 25.25 | 43.96 | 34.47 | 57.68 | 24.10 | 38.75 |
| e5-base-v2 | 55.91 | 37.26 | 39.72 | 30.50 | 19.57 | 57.80 | 40.13 | 69.27 | 5.11 | 38.17 |
| e5-large-v2 | 57.58 | 37.70 | 40.81 | 31.38 | 19.88 | 58.53 | 40.98 | 67.54 | 5.22 | 37.91 |
| functionwordfreq | 51.00 | 35.58 | 43.56 | 31.56 | 26.36 | 53.12 | 40.20 | 63.21 | 9.52 | 37.64 |
| tfidfngrams_mud_subset_1-2grams.pkl | 49.91 | 33.96 | 36.19 | 31.67 | 27.14 | 57.98 | 39.48 | 66.90 | 5.66 | 37.35 |
| tfidfngrams_fineweb_sample10bt_1-2grams.pkl | 49.19 | 33.79 | 36.20 | 31.70 | 25.34 | 58.70 | 39.15 | 67.13 | 5.55 | 37.28 |
| lisa_checkpoint | 56.89 | 37.95 | 37.38 | 32.95 | 22.69 | 54.00 | 40.31 | 64.02 | 7.10 | 37.15 |
| tfidfngrams_mud_subset_1-3grams.pkl | 48.90 | 33.75 | 36.37 | 31.84 | 22.90 | 58.05 | 38.63 | 66.55 | 5.62 | 36.94 |
| gte-base-en-v1.5 | 53.62 | 36.51 | 39.44 | 31.76 | 19.05 | 56.56 | 39.49 | 65.77 | 5.31 | 36.85 |
| tfidfngrams_fineweb_sample10bt_1-3grams.pkl | 48.49 | 33.55 | 36.30 | 31.75 | 21.84 | 57.64 | 38.26 | 66.68 | 5.47 | 36.81 |
| bge-large-en-v1.5 | 56.15 | 37.31 | 38.23 | 29.71 | 18.22 | 57.35 | 39.50 | 65.54 | 5.28 | 36.77 |
| bge-base-en-v1.5 | 54.69 | 36.67 | 36.96 | 30.75 | 18.16 | 58.25 | 39.25 | 65.79 | 5.22 | 36.75 |
| jina-embeddings-v3 | 56.24 | 37.51 | 39.77 | 31.10 | 18.63 | 57.25 | 40.08 | 63.80 | 5.53 | 36.47 |
| gte-large-en-v1.5 | 57.09 | 36.67 | 41.20 | 31.38 | 19.79 | 57.26 | 40.57 | 63.40 | 5.45 | 36.47 |
| all-mpnet-base-v2 | 56.07 | 37.55 | 39.03 | 31.47 | 18.39 | 57.50 | 40.00 | 62.18 | 5.06 | 35.75 |
| neurobiber | 47.31 | 32.38 | 34.49 | 31.46 | 25.00 | 34.04 | 34.11 | 64.11 | 7.99 | 35.40 |
