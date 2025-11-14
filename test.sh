#!/bin/sh
export CUDA_VISIBLE_DEVICES=0;

set -e
set -u

pip install -e .

steb clustering --list-datasets

steb clustering "rrivera1849/LUAR-MUD" --dataset "sms_spam" --force-reload -e 1
