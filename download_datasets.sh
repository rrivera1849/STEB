#!/bin/sh
mkdir -p ./raw_datasets
cd ./raw_datasets

# Clean up directories from previous runs to ensure idempotency
rm -rf hate-speech-dataset
rm -rf hate-speech-and-offensive-language
rm -rf CharCnn_Keras
rm -rf enron_authorship_corpus
rm -rf 'Enron (80 authors)'
rm -rf STEL
rm -f blogs.zip
rm -f n77w7mygwg-1.zip
rm -f Enron-Authorship-Verification-Corpus.zip

git clone --depth 1 https://github.com/Vicomtech/hate-speech-dataset.git
rm -rf hate-speech-dataset/.git
git clone --depth 1 https://github.com/t-davidson/hate-speech-and-offensive-language.git
rm -rf hate-speech-and-offensive-language/.git
git clone --depth 1 https://github.com/mhjabreel/CharCnn_Keras.git
rm -rf CharCnn_Keras/.git

# Note: The blog_authorship_corpus download is currently disabled due to an unreliable server.
# curl -k -O http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip

curl -k -O https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n77w7mygwg-1.zip
unzip n77w7mygwg-1.zip
unzip Enron-Authorship-Verification-Corpus.zip
rm n77w7mygwg-1.zip
rm Enron-Authorship-Verification-Corpus.zip
mv 'Enron (80 authors)' enron_authorship_corpus

gdown https://drive.google.com/file/d/17eLjiDLhXXtrfqC1FzToHKrwTcwsAolN/view?usp=sharing --fuzzy
tar zxvf jigsaw-toxic-comment-classification-challenge.tar.gz
rm jigsaw-toxic-comment-classification-challenge.tar.gz

mkdir ./dummy_retrieval
cd ./dummy_retrieval
gdown https://drive.google.com/file/d/17HkJs9E5nnwuDfONiUy3sw0-Nj53SwvL/view?usp=sharing --fuzzy
cd ..

mkdir fanfiction_retrieval
cd fanfiction_retrieval
gdown https://drive.google.com/file/d/1imzUA2kg4782WnYWxlKICyk1MmPCCtsV/view?usp=sharing --fuzzy
tar zxvf fanfiction_retrieval.tar.gz
rm fanfiction_retrieval.tar.gz
cd ..

mkdir amazon_retrieval
cd amazon_retrieval
gdown https://drive.google.com/file/d/1HFWoFk3V7vqM_DbZzd8ixecdbjHp29c4/view?usp=sharing --fuzzy
tar zxvf amazon_retrieval.tar.gz
rm amazon_retrieval.tar.gz
cd ..

git clone --depth 1 --filter=blob:none --sparse https://github.com/nlpsoc/STEL.git STEL
cd STEL
git sparse-checkout set Data/STEL
rm -rf .git
cd ..

# PAN AV-15
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

# PAN AV-14
mkdir pan14-authorship-verification-test-2014-04-22
cd pan14-authorship-verification-test-2014-04-22
wget https://zenodo.org/records/3716033/files/pan14-authorship-verification-test-and-training.zip
unzip pan14-authorship-verification-test-and-training.zip
unzip pan14-authorship-verification-test-corpus1-2014-04-22.zip
unzip pan14-authorship-verification-test-corpus2-2014-04-22.zip
unzip -o "*test-corpus1*.zip"
unzip -o "*test-corpus2*.zip"
rm *.zip

# PAN AV-13
wget https://zenodo.org/records/3715999/files/pan13-authorship-verification-test-and-training.zip
unzip pan13-authorship-verification-test-and-training.zip
unzip pan13-authorship-verification-test-corpus2-2013-05-29.zip
rm *.zip
