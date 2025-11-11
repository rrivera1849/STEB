#!/bin/sh
export CUDA_VISIBLE_DEVICES=0;

set -e
set -u

DATASETS=(
    20_Newsgroups_Fixed
    ag_news
    corpus-of-diverse-styles
    emotion
    enron_spam
    financial_phrasebank
    hate_speech
    hate_speech_and_offensive_language
    jigsaw_toxicity_pred
    reuters21578
    sms_spam
    telegram-spam-ham
    twitter-airline-sentiment
    yelp_polarity
)

for dataset_name in ${DATASETS[@]};
do
    python main.py \
        --dataset ${dataset_name} \
        --model_name_or_path "rrivera1849/LUAR-MUD" \
        --force_reload \
        -e 1
done
