#!/bin/sh
mkdir ./data
cd ./data
git clone --depth 1 https://github.com/Vicomtech/hate-speech-dataset.git
rm -rf hate-speech-dataset/.git
git clone --depth 1 https://github.com/t-davidson/hate-speech-and-offensive-language.git
rm -rf hate-speech-and-offensive-language/.git
git clone --depth 1 https://github.com/mhjabreel/CharCnn_Keras.git
rm -rf CharCnn_Keras/.git
curl -k -O http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip
curl -k -O https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n77w7mygwg-1.zip
unzip n77w7mygwg-1.zip
unzip Enron-Authorship-Verification-Corpus.zip
rm n77w7mygwg-1.zip
rm Enron-Authorship-Verification-Corpus.zip
mv 'Enron (80 authors)' enron_authorship_corpus
