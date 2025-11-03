#!/bin/sh
export CUDA_VISIBLE_DEVICES=0;

set -e
set -u

DATASETS=(
    "corpus-of-diverse-styles"
)

# Run LUAR model
python main.py \
    --dataset "corpus-of-diverse-styles" \
    --model_name_or_path "rrivera1849/LUAR-MUD" \
    --model_type luar \
    --force_reload \
    -e 1

# Run Style-Embedding model
python main.py \
    --dataset "corpus-of-diverse-styles" \
    --model_name_or_path "AnnaWegmann/Style-Embedding" \
    --model_type hf \
    --force_reload \
    -e 1

