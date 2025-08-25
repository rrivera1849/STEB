
# STEB: Style Text Embedding Benchmark

Measuring the quality of Style Text Embeddings across various axes. 

High-level TODOs:
- Details on how evaluations are performed, perhaps with an image
- Make it into a package like mteb?
- Create the leaderboard
- Add probing for features (code for this exists in RRS machine but it's extremely hacky atm)

## Installing Requirements

After creating a virtual environment, download and install the requirements as follows:
```bash
pip install -r requirements.txt
python spacy -m download en_core_web_sm
```
## Downloading Datasets

The following will download datasets not available through HF into the "./datasets" folder:

```bash
./prepare_datasets.sh
```

Results are stored in "./outputs".

## Running a Debug Evaluation

To run a debug evaluation, run the following:
```bash
./debug.sh
```

## Running General Evaluations

```bash
python main.py \
    --dataset <DATASET_NAME> \
    --model_name_or_path <HF_ID_OR_PATH> \
    -e <NUMBER_OF_SAMPLES_PER_EMBEDDING>
```

## Corpora Available

The following corpora are available:
* reuters21578
* billray110/corpus-of-diverse-styles
* jigsaw_toxicity_pred
* emotion
* ag_news
* rungalileo/20_Newsgroups_Fixed
* financial_phrasebank
* osanseviero/twitter-airline-sentiment
* blog_authorship_corpus
* sms_spam
* SetFit/enron_spam
* thehamkercat/telegram-spam-ham
* yelp_polarity
* hate_speech
* hate_speech_and_offensive_language
* enron_authorship_corpus