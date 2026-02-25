#!/bin/sh

# Usage: ./download_datasets.sh [--purge]
#   --purge    Remove all existing datasets before downloading

if [ "$1" = "--purge" ]; then
    echo "Purging all datasets in ./raw_datasets..."
    rm -rf ./raw_datasets
fi

mkdir -p ./raw_datasets
cd ./raw_datasets

# Note: The blog_authorship_corpus download is currently disabled due to an unreliable server.
# curl -k -O http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip

if [ ! -d "hate-speech-dataset" ]; then
    echo "Downloading hate-speech-dataset..."
    git clone --depth 1 https://github.com/Vicomtech/hate-speech-dataset.git
    rm -rf hate-speech-dataset/.git
else
    echo "Skipping hate-speech-dataset (already exists)"
fi

if [ ! -d "hate-speech-and-offensive-language" ]; then
    echo "Downloading hate-speech-and-offensive-language..."
    git clone --depth 1 https://github.com/t-davidson/hate-speech-and-offensive-language.git
    rm -rf hate-speech-and-offensive-language/.git
else
    echo "Skipping hate-speech-and-offensive-language (already exists)"
fi

if [ ! -d "CharCnn_Keras" ]; then
    echo "Downloading CharCnn_Keras..."
    git clone --depth 1 https://github.com/mhjabreel/CharCnn_Keras.git
    rm -rf CharCnn_Keras/.git
else
    echo "Skipping CharCnn_Keras (already exists)"
fi

if [ ! -d "enron_authorship_corpus" ]; then
    echo "Downloading enron_authorship_corpus..."
    curl -O https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n77w7mygwg-1.zip
    unzip n77w7mygwg-1.zip
    unzip Enron-Authorship-Verification-Corpus.zip
    rm n77w7mygwg-1.zip
    rm Enron-Authorship-Verification-Corpus.zip
    mv 'Enron (80 authors)' enron_authorship_corpus
else
    echo "Skipping enron_authorship_corpus (already exists)"
fi

if [ ! -d "jigsaw-toxic-comment-classification-challenge" ]; then
    echo "Downloading jigsaw-toxic-comment-classification-challenge..."
    gdown https://drive.google.com/file/d/17eLjiDLhXXtrfqC1FzToHKrwTcwsAolN/view?usp=sharing --fuzzy
    tar zxvf jigsaw-toxic-comment-classification-challenge.tar.gz
    rm jigsaw-toxic-comment-classification-challenge.tar.gz
else
    echo "Skipping jigsaw-toxic-comment-classification-challenge (already exists)"
fi

if [ ! -d "dummy_retrieval" ]; then
    echo "Downloading dummy_retrieval..."
    mkdir ./dummy_retrieval
    cd ./dummy_retrieval
    gdown https://drive.google.com/file/d/17HkJs9E5nnwuDfONiUy3sw0-Nj53SwvL/view?usp=sharing --fuzzy
    cd ..
else
    echo "Skipping dummy_retrieval (already exists)"
fi

if [ ! -d "fanfiction_retrieval" ]; then
    echo "Downloading fanfiction_retrieval..."
    mkdir fanfiction_retrieval
    cd fanfiction_retrieval
    gdown https://drive.google.com/file/d/1imzUA2kg4782WnYWxlKICyk1MmPCCtsV/view?usp=sharing --fuzzy
    tar zxvf fanfiction_retrieval.tar.gz
    rm fanfiction_retrieval.tar.gz
    cd ..
else
    echo "Skipping fanfiction_retrieval (already exists)"
fi

if [ ! -d "amazon_retrieval" ]; then
    echo "Downloading amazon_retrieval..."
    mkdir amazon_retrieval
    cd amazon_retrieval
    gdown https://drive.google.com/file/d/1HFWoFk3V7vqM_DbZzd8ixecdbjHp29c4/view?usp=sharing --fuzzy
    tar zxvf amazon_retrieval.tar.gz
    rm amazon_retrieval.tar.gz
    cd ..
else
    echo "Skipping amazon_retrieval (already exists)"
fi

# Graded Formality GPT-5-mini
if [ ! -d "graded_formality" ]; then
    echo "Downloading graded_formality..."
    mkdir -p graded_formality
    cd graded_formality
    gdown https://drive.google.com/file/d/1TKtgg-6j2Yd-GTfiq0nnHRPcZC-VpGnu/view?usp=drive_link --fuzzy
    unzip graded_formality_generated_gpt-5-mini.zip
    rm graded_formality_generated_gpt-5-mini.zip
    rm -rf __MACOSX
    cd ..
else
    echo "Skipping graded_formality (already exists)"
fi

if [ ! -d "STEL" ]; then
    echo "Downloading STEL..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/nlpsoc/STEL.git STEL
    cd STEL
    git sparse-checkout set Data/STEL
    rm -rf .git
    cd ..
else
    echo "Skipping STEL (already exists)"
fi

# OneStopEnglishCorpus (Texts-SeparatedByReadingLevel)
if [ ! -d "OneStopEnglishCorpus" ]; then
    echo "Downloading OneStopEnglishCorpus..."
    git clone --depth 1 https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git OneStopEnglishCorpus
    rm -rf OneStopEnglishCorpus/.git
else
    echo "Skipping OneStopEnglishCorpus (already exists)"
fi

# PAN AV-15
if [ ! -d "pan15-authorship-verification-test-dataset2-2015-04-19" ]; then
    echo "Downloading PAN AV-15..."
    wget https://zenodo.org/records/3737563/files/pan15-authorship-verification-test-and-training.zip
    unzip pan15-authorship-verification-test-and-training.zip
    unzip pan15-authorship-verification-test-dataset2-2015-04-19.zip
    rm pan15-authorship-verification-test-and-training.zip
    rm pan15-authorship-verification-test-dataset2-2015-04-19.zip
    rm pan15-authorship-verification-training-dataset-2015-04-19.zip
    cd pan15-authorship-verification-test-dataset2-2015-04-19
    unzip "*.zip"
    rm *.zip
    cd ../
else
    echo "Skipping PAN AV-15 (already exists)"
fi

# PAN AV-14
if [ ! -d "pan14-authorship-verification-test-2014-04-22" ]; then
    echo "Downloading PAN AV-14..."
    mkdir pan14-authorship-verification-test-2014-04-22
    cd pan14-authorship-verification-test-2014-04-22
    wget https://zenodo.org/records/3716033/files/pan14-authorship-verification-test-and-training.zip
    unzip pan14-authorship-verification-test-and-training.zip
    unzip pan14-authorship-verification-test-corpus1-2014-04-22.zip
    unzip pan14-authorship-verification-test-corpus2-2014-04-22.zip
    unzip -o "*test-corpus1*.zip"
    unzip -o "*test-corpus2*.zip"
    rm *.zip
    cd ../
else
    echo "Skipping PAN AV-14 (already exists)"
fi

# PAN AV-13
if [ ! -d "pan13-authorship-verification-test-corpus2-2013-05-29" ]; then
    echo "Downloading PAN AV-13..."
    wget https://zenodo.org/records/3715999/files/pan13-authorship-verification-test-and-training.zip
    unzip pan13-authorship-verification-test-and-training.zip
    unzip pan13-authorship-verification-test-corpus2-2013-05-29.zip
    rm *.zip
else
    echo "Skipping PAN AV-13 (already exists)"
fi

# GEDE Essay Detection
if [ ! -d "gede_essay_detection" ]; then
    echo "Downloading gede_essay_detection..."
    mkdir -p gede_essay_detection
    gdown https://drive.google.com/file/d/1c3x_CR44ZCUqHf1dHVPm7K04ZIbTSYoD/view?usp=drive_link --fuzzy
    tar -zxvf gede_essay_detection.tar.gz
    rm gede_essay_detection.tar.gz
else
    echo "Skipping gede_essay_detection (already exists)"
fi

# CORE Register Corpus
if [ ! -d "CORE-corpus" ]; then
    echo "Downloading CORE-corpus..."
    mkdir -p CORE-corpus
    cd CORE-corpus
    wget https://github.com/TurkuNLP/CORE-corpus/raw/master/train.tsv.gz
    wget https://github.com/TurkuNLP/CORE-corpus/raw/master/dev.tsv.gz
    wget https://github.com/TurkuNLP/CORE-corpus/raw/master/test.tsv.gz
    gunzip train.tsv.gz dev.tsv.gz test.tsv.gz
    cd ..
else
    echo "Skipping CORE-corpus (already exists)"
fi

# RadioTalk
if [ ! -d "radiotalk" ]; then
    echo "Downloading radiotalk..."
    mkdir -p radiotalk
    cd radiotalk
    gdown https://drive.google.com/file/d/1NHgHZDlMB9Yh9SGQdSTjz_CNLZZqGCuW/view?usp=drive_link --fuzzy
    tar -zxvf radiotalk.tar.gz
    rm radiotalk.tar.gz
    cd ..
else
    echo "Skipping radiotalk (already exists)"
fi

echo "Done downloading datasets."
