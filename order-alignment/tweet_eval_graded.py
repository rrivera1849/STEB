"""
    Prompting LLMs to make TweetEval tweets and SPAM SMS messages more and more formal.
"""

from datasets import load_dataset
from typing import List
from openai import OpenAI


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


def progressive_formalization(text: str, client: OpenAI, model: str = "gpt-5.2") -> List[str]:
    """
    Progressively formalize text through multiple iterations.

    Args:
        text: The original informal text to formalize
        client: OpenAI client instance
        model: The model to use (default: gpt-5.2)

    Returns:
        List of 5 strings: [original, level1, level2, level3, level4, level5]
        where each level is progressively more formal
    """
    results = [text]  # Start with the original text

    # First prompt: initial formalization
    messages = [
        {"role": "user", "content": f"Directly return the answer. Make the following utterance a bit more formal: {text}"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    formalized_text = response.choices[0].message.content
    results.append(formalized_text)

    # Add the assistant's response to the conversation history
    messages.append({"role": "assistant", "content": formalized_text})

    # Four additional iterations with "More formal"
    for _ in range(4):
        messages.append({"role": "user", "content": "More formal"})

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        formalized_text = response.choices[0].message.content
        results.append(formalized_text)

        # Add the assistant's response to continue the conversation
        messages.append({"role": "assistant", "content": formalized_text})

    return results


def main():
    """Main function to generate progressively formalized versions of tweets and SMS."""
    # Initialize OpenAI client
    # client = OpenAI()

    # Load datasets
    print("Loading datasets...")
    tweets = get_informal_tweet_eval_tweets()
    sms = get_informal_sms_messages()
    print(f"Loaded {len(tweets)} tweets and {len(sms)} SMS messages")

    # Example: Process first tweet
    if tweets:
        print("\nExample: Progressive formalization of first tweet")
        print(f"Original: {tweets[0]}")
        print("\nGenerating progressive formalizations...")

        formalized_versions = progressive_formalization(tweets[0], client)

        for i, version in enumerate(formalized_versions):
            print(f"[Level {i}]: {version}")


if __name__ == "__main__":
    main()