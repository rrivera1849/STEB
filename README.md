
# STEB: Style Text Embedding Benchmark

Measuring the quality of Style Text Embeddings across various axes. 

High-level TODOs:
- Details on how evaluations are performed, perhaps with an image
- Make it into a package like mteb
- Create the leaderboard
- Add probing for features (code for this exists in RRS machine but it's extremely hacky atm)

## Installing Requirements

After creating a virtual environment, download and install the requirements as follows:
```bash
pip install -r requirements.txt
python spacy -m download en_core_web_sm
```
## Downloading Datasets

**Note:** The `jigsaw_toxicity_pred` dataset is not downloaded automatically. You will need to download it manually from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place it in the `steb_datasets` directory.

The following will download datasets not available through HF into the "./steb_datasets" folder:

```bash
./prepare_datasets.sh
```

## Running a Debug Evaluation

To run a debug evaluation, run the following:
```bash
./debug.sh
```
Results are stored in "./outputs".


## Running General Evaluations

```bash
python main.py \
    --dataset <DATASET_NAME> \
    --model_name_or_path <HF_ID_OR_PATH> \
    --model_type <MODEL_TYPE> \
    -e <NUMBER_OF_SAMPLES_PER_EMBEDDING>
```

## Adding a New Model

To add a new model, you need to:

1.  Create a new Python file in the `models` directory (e.g., `models/my_model.py`).
2.  In this file, create a class that inherits from `STEBModel` (from `models.base`) and implements the `embed_single` and `embed_multiple` methods.
3.  Register your new model in `models/__init__.py` by adding it to the `MODEL_REGISTRY` dictionary.

## Adding a New Dataset

To add a new dataset, you need to:

1.  Create a new subdirectory in the `steb_datasets` directory with the name of your dataset (e.g., `steb_datasets/my_dataset`).
2.  Inside this new subdirectory, create a `config.json` file.
3.  This `config.json` file should contain the following keys:
    *   `dataset_name`: The name of the dataset.
    *   `type`: The type of the dataset, either `"huggingface"` or `"custom"`.
    *   `record_handler`: Specifies how to extract the text and label from a dataset record. It should have `text_getter` and `label_getter` keys.
    *   If the `type` is `"huggingface"`, you must include `loader_kwargs`: A dictionary of arguments that will be passed to the `load_dataset` function from the Hugging Face `datasets` library.
    *   If the `type` is `"custom"`, you must include `data_dir`: The path to the dataset's data directory. You will also need to create a `loader.py` file in the same directory and specify the loader function in the `config.json` with the `loader_function` key.
    *   If your dataset requires a custom label transformation, you can add the function to the `loader.py` file and specify it in the `config.json` with the `label_getter_function` key.

Your new dataset will be automatically discovered and made available as a choice for the `--dataset` argument.

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
* sms_spam
* SetFit/enron_spam
* thehamkercat/telegram-spam-ham
* yelp_polarity
* hate_speech
* hate_speech_and_offensive_language
* enron_authorship_corpus
