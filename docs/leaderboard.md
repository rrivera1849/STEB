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
| LUSR | *20.86* | 61.37 | 68.03 | 33.87 | 65.37 | *58.48* | **51.33** |
| star | **24.95** | **64.33** | **72.71** | 23.93 | 66.96 | 49.86 | *50.46* |
| LUAR-CRUD | 19.39 | 60.77 | 70.98 | 19.80 | **77.64** | 53.75 | 50.39 |
| LUAR-MUD | 19.79 | 62.09 | 70.12 | 20.81 | *75.67* | 53.04 | 50.25 |
| styledistance | 16.70 | 59.56 | 65.77 | **44.02** | 49.61 | 57.87 | 48.92 |
| deberta-v3-large | 20.30 | 59.10 | *71.68* | 24.23 | 63.24 | 53.41 | 48.66 |
| multilingual-style-representation | 18.57 | *64.18* | 69.66 | 21.85 | 63.49 | 51.01 | 48.13 |
| roberta-large | 15.33 | 60.77 | 68.54 | 20.71 | 61.07 | 57.92 | 47.39 |
| bert-large-cased | 14.23 | 60.71 | 66.60 | 21.57 | 58.88 | 55.16 | 46.19 |
| roberta-base | 14.26 | 60.65 | 66.16 | 22.47 | 50.63 | **61.32** | 45.91 |
| multilingual-style-representation-Llama-3.2 | 19.53 | 63.01 | 67.05 | 19.90 | 55.57 | 49.19 | 45.71 |
| deberta-v3-base | 16.85 | 58.05 | 67.94 | 23.41 | 55.17 | 51.73 | 45.52 |
| bert-large-uncased | 12.23 | 60.17 | 66.70 | 21.18 | 60.23 | 52.48 | 45.50 |
| Style-Embedding | 13.45 | 59.43 | 65.26 | 33.81 | 45.87 | 52.54 | 45.06 |
| ModernBERT-large | 17.39 | 59.91 | 64.56 | 20.56 | 52.17 | 54.06 | 44.77 |
| ModernBERT-base | 14.81 | 59.03 | 63.68 | 19.20 | 51.25 | 53.91 | 43.65 |
| styledistance_synthetic_only | 12.34 | 56.92 | 58.40 | *42.76* | 31.53 | 56.69 | 43.11 |
| e5-large-v2 | 13.34 | 60.01 | 63.85 | 15.82 | 49.91 | 49.03 | 41.99 |
| gte-large-en-v1.5 | 13.54 | 60.81 | 62.04 | 16.20 | 49.60 | 48.87 | 41.84 |
| e5-base-v2 | 11.56 | 60.26 | 61.96 | 15.26 | 49.62 | 52.13 | 41.80 |
| tfidfngrams_fineweb_sample10bt_1-2grams.pkl | 6.46 | 56.98 | 65.04 | 13.72 | 50.95 | 55.74 | 41.48 |
| bge-base-en-v1.5 | 9.84 | 59.24 | 62.33 | 15.82 | 51.44 | 49.45 | 41.35 |
| tfidfngrams_mud_subset_1-2grams.pkl | 6.58 | 56.72 | 65.12 | 13.84 | 49.77 | 55.54 | 41.26 |
| gte-base-en-v1.5 | 12.98 | 59.89 | 58.93 | 15.62 | 49.36 | 50.64 | 41.24 |
| tfidfngrams_mud_subset_1-3grams.pkl | 6.34 | 56.46 | 64.96 | 13.84 | 50.26 | 55.34 | 41.20 |
| surface_pos.yaml | 10.47 | 56.75 | 60.92 | 25.87 | 39.52 | 53.30 | 41.14 |
| tfidfngrams_fineweb_sample10bt_1-3grams.pkl | 6.17 | 56.76 | 65.08 | 13.67 | 49.00 | 55.51 | 41.03 |
| jina-embeddings-v3 | 10.08 | 60.36 | 63.83 | 17.11 | 47.35 | 45.62 | 40.73 |
| bge-large-en-v1.5 | 9.13 | 59.69 | 63.25 | 16.23 | 48.79 | 47.23 | 40.72 |
| all-mpnet-base-v2 | 8.52 | 60.18 | 63.22 | 14.62 | 48.99 | 45.64 | 40.19 |
| lisa_checkpoint | 13.82 | 60.73 | 60.46 | 16.48 | 42.96 | 46.38 | 40.14 |
| opt-1.3b | 3.95 | 52.32 | 56.24 | 24.08 | 39.07 | 45.82 | 36.91 |
| Qwen3.5-4B-Base | 4.35 | 52.74 | 57.97 | 24.88 | 37.15 | 43.02 | 36.69 |
| functionwordfreq | 6.69 | 55.14 | 61.86 | 14.51 | 43.14 | 37.98 | 36.55 |
| Qwen3.5-2B-Base | 3.68 | 52.40 | 57.38 | 24.61 | 35.13 | 42.61 | 35.97 |
| Qwen3.5-0.8B-Base | 3.96 | 52.65 | 57.62 | 25.07 | 32.83 | 43.30 | 35.91 |
| gpt2-xl | 4.23 | 51.63 | 53.98 | 26.32 | 28.82 | 48.66 | 35.61 |
| mstyledistance | 7.22 | 53.71 | 54.16 | 31.38 | 20.04 | 45.96 | 35.41 |
| Qwen3-Embedding-8B | 2.96 | 51.58 | 52.11 | 23.11 | 35.17 | 42.96 | 34.65 |
| Qwen2-0.5B | 3.63 | 51.53 | 52.04 | 25.51 | 25.70 | 42.34 | 33.46 |
| Qwen3-0.6B-Base | 3.60 | 51.37 | 53.08 | 26.01 | 23.67 | 41.78 | 33.25 |
| neurobiber | 8.89 | 54.91 | 56.54 | 14.13 | 9.06 | 51.64 | 32.53 |

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
| multilingual-style-representation-Llama-3.2 | 67.05 | 40.69 | 42.49 | 34.80 | 35.84 | 63.12 | 47.33 | 67.89 | 10.07 | 41.77 |
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
| gte-large-en-v1.5 | 57.09 | 36.67 | 41.20 | 31.38 | 19.79 | 57.26 | 40.57 | 63.40 | 5.45 | 36.47 |
| jina-embeddings-v3 | 56.24 | 37.51 | 39.77 | 31.10 | 18.63 | 57.25 | 40.08 | 63.80 | 5.53 | 36.47 |
| all-mpnet-base-v2 | 56.07 | 37.55 | 39.03 | 31.47 | 18.39 | 57.50 | 40.00 | 62.18 | 5.06 | 35.75 |
| neurobiber | 47.31 | 32.38 | 34.49 | 31.46 | 25.00 | 34.04 | 34.11 | 64.11 | 7.99 | 35.40 |
