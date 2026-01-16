"""
Prompt OpenAI chat models to progressively formalize utterances.
"""
import argparse
import csv
import os
from datetime import datetime, timezone

from openai import OpenAI

from informal_human_data import get_informal_tweet_eval_tweets, get_informal_sms_messages


PROMPT = ("Gradually increase the formality of the following utterance in a small increment. "
          "Whenever I ask for a more formal version, adjust the formality upward only slightly—just one level at a time. "
          "Respond only with the revised sentence and nothing else.")
FOLLOW_UP = "More formal"

TEST_SENTENCES = [
    "Oh k...i'm watching here:)",
    "K..k:)where are you?how did you performed?",
    "Going for dinner.msg you after.",
    "i see. When we finish we have loads of loans to pay",
    "@user GOD DAMN IT... YOU ARE SO RIGHT. THERE'S NO GOING WRONG WITH THIS STUFF.",
    "@user @user I didn't know Oprah Winfrey knew how to play tennis. .. 😱",
    "@user She is so cute ♡",
]

MODELS = [
    "gpt-5.1-chat-latest",
    "gpt-5-nano-2025-08-07",
    "gpt-5-mini-2025-08-07",
]

N_LEVELS = 4


def formalize_progressively(client, model: str, text: str, max_new_tokens: int, temperature: float) -> list[str]:
    """Returns [original, level1, level2, level3, level4]"""
    results = [text]
    messages = [
        {"role": "user", "content": f"{PROMPT}\n\n{text}"}
    ]

    for i in range(N_LEVELS):
        if i > 0:
            messages.append({"role": "user", "content": FOLLOW_UP})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_completion_tokens=max_new_tokens,
            seed=42,
        )
        assistant_msg = response.choices[0].message.content.strip()
        results.append(assistant_msg)
        messages.append({"role": "assistant", "content": assistant_msg})

    return results


def save_results(output_path: str, rows: list[dict]):
    fieldnames = ["model", "source", "input", "level1", "level2", "level3", "level4"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=os.path.join("order-alignment", "graded_closed_models"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-run", action="store_true", help="Run on full dataset instead of test sentences")
    parser.add_argument("--model", type=str, help="Model to run (required for --full-run, ignored otherwise)")
    args = parser.parse_args()

    if args.full_run and not args.model:
        parser.error("--model is required when using --full-run")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    if args.full_run:
        tweets = get_informal_tweet_eval_tweets()
        sms_messages = get_informal_sms_messages()
        sentences = [(text, "sms") for text in sms_messages] + [(text, "tweet") for text in tweets]
        models = [args.model]
        print(f"Full run: loaded {len(sms_messages)} SMS messages and {len(tweets)} tweets")
    else:
        sentences = [(text, "test") for text in TEST_SENTENCES]
        models = MODELS
        print(f"Test run: using {len(sentences)} test sentences")

    client = OpenAI()

    for model in models:
        output_path = os.path.join(output_dir, f"{model.replace('/', '_')}.csv")
        rows = []

        for i, (text, source) in enumerate(sentences):
            print(f"[{model}] Processing {i+1}/{len(sentences)}: {text[:30]}...")
            try:
                levels = formalize_progressively(client, model, text, args.max_new_tokens, args.temperature)
                rows.append({
                    "model": model,
                    "source": source,
                    "input": text,
                    "level1": levels[1],
                    "level2": levels[2],
                    "level3": levels[3],
                    "level4": levels[4],
                })
            except Exception as e:
                print(f"  Error: {e}")
                rows.append({
                    "model": model,
                    "source": source,
                    "input": text,
                    "level1": f"ERROR: {e}",
                    "level2": "",
                    "level3": "",
                    "level4": "",
                })
            save_results(output_path, rows)

        print(f"Done: {output_path}")


if __name__ == "__main__":
    main()