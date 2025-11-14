#!/bin/sh
export CUDA_VISIBLE_DEVICES=0;

set -e
set -u

pip install -e .

steb run \
    -t "sms_spam" \
    -m "rrivera1849/LUAR-MUD" \
    --force-reload \
    -e 1
