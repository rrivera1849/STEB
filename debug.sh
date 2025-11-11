#!/bin/sh
export CUDA_VISIBLE_DEVICES=0;

set -e
set -u

python main.py \
    --dataset "jigsaw_toxicity_pred" \
    --model_name_or_path "rrivera1849/LUAR-MUD" \
    --force_reload \
    -e 1
