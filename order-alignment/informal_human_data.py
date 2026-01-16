
"""
    get Tweet and SMS data
"""
from datasets import load_dataset
from typing import List

def get_informal_tweet_eval_tweets() -> List[str]:
    """
        Returns: the non-offensive messages (label=0) from the train split of the offensive subset of the TweetEval
            at https://huggingface.co/datasets/cardiffnlp/tweet_eval

    """
    huggingface_path = "cardiffnlp/tweet_eval"

    # Load the offensive subset of TweetEval
    dataset = load_dataset(huggingface_path, "offensive", split="train")

    # Filter for non-offensive tweets (label=0)
    non_offensive_tweets = [
        example["text"] for example in dataset if example["label"] == 0
    ]

    return non_offensive_tweets


def get_informal_sms_messages() -> List[str]:
    """
    Returns: the non-spam messages (label=0) from the train split of the SMS Spam Collection Dataset
            at https://huggingface.co/datasets/ucirvine/sms_spam

    """
    huggingface_path = "ucirvine/sms_spam"

    # Load the SMS spam dataset
    dataset = load_dataset(huggingface_path, split="train")

    # Filter for non-spam messages (label=0, which is "ham")
    non_spam_messages = [
        example["sms"].rstrip() for example in dataset if example["label"] == 0
    ]

    return non_spam_messages
