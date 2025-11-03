#!/bin/sh
export CUDA_VISIBLE_DEVICES=0;

set -e
set -u

DATASETS=(
    "corpus-of-diverse-styles"
)
MODELS=(
    "rrivera1849/LUAR-MUD"
    "AnnaWegmann/Style-Embedding"
)
for model in ${MODELS[@]}; do
    for dataset in ${DATASETS[@]}; do
        python main.py \
            --dataset ${dataset} \
            --model_name_or_path ${model} \
            --force_reload \
            -e 1
    done
done
