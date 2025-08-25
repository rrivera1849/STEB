
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
CACHE_DIR = config["Application_Paths"]["cache_dir"]
PROCESSED_DATA_DIR = config["Application_Paths"]["processed_dataset_dir"]

BENCHMARK_GROUPS = {
    "Topic Classification": [
        "reuters21578",
        "ag_news",
        "20_Newsgroups_Fixed",
    ],
    "Style Classification": [
        "corpus-of-diverse-styles",
        "blog_authorship_corpus",
        "enron_authorship_corpus",
    ],
    "Toxicity Classification": [
        "jigsaw_toxicity_pred",
        "hate_speech",
        "hate_speech_and_offensive_language",
    ],
    "Sentiment Analysis": [
        "yelp_polarity",
        "financial_phrasebank",
        "twitter-airline-sentiment",
        "emotion",
    ],
    "Spam Detection": [
        "sms_spam",
        "enron_spam",
        "telegram-spam-ham",
    ],
}