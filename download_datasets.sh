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
rm -rf CharCnn_Keraalso as/.git

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

git clone --depth 1 --filter=blob:none --sparse https://github.com/nlpsoc/STEL.git STEL
cd STEL
git sparse-checkout set Data/STEL
rm -rf .git
cd ..