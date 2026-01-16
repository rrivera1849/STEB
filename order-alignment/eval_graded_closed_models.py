"""
Prompt OpenAI chat models to progressively formalize utterances.
"""
import json
import os
from datetime import datetime, timezone

from openai import OpenAI


SENTENCES = [
    "Oh k...i'm watching here:)",
    "K..k:)where are you?how did you performed?",
    "Going for dinner.msg you after.",
    "i see. When we finish we have loads of loans to pay",
    "@user GOD DAMN IT... YOU ARE SO RIGHT. THERE'S NO GOING WRONG WITH THIS STUFF.",
    "@user @user I didn't know Oprah Winfrey knew how to play tennis. .. 😱",
    "@user She is so cute ♡",
]

MODELS = [
    # "gpt-5.1-chat-latest",
    # "gpt-5-nano-2025-08-07",
    "gpt-5-mini-2025-08-07",
]

N_LEVELS = 4


def formalize_progressively(client, model: str, text: str) -> list[str]:
    """Returns [original, level1, level2, level3, level4]"""
    results = [text]
    messages = [
        {"role": "user", "content": f"Directly return the answer. Make the following utterance a bit more formal: {text}"}
    ]

    for i in range(N_LEVELS):
        if i > 0:
            messages.append({"role": "user", "content": "A bit more formal"})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            seed=42,
        )
        assistant_msg = response.choices[0].message.content.strip()
        results.append(assistant_msg)
        messages.append({"role": "assistant", "content": assistant_msg})

    return results

def save_results(output_path: str, model: str, results: list):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "results": results}, f, ensure_ascii=False, indent=2)


def main():
    client = OpenAI()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = f"order-alignment/graded_closed_models/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    for model in MODELS:
        output_path = os.path.join(output_dir, f"{model.replace('/', '_')}.json")
        results = []

        for i, sentence in enumerate(SENTENCES):
            print(f"[{model}] Processing {i+1}/{len(SENTENCES)}: {sentence[:30]}...")
            levels = formalize_progressively(client, model, sentence)
            results.append({
                "input": sentence,
                "levels": levels,
            })
            save_results(output_path, model, results)

        print(f"Done: {output_path}")


if __name__ == "__main__":
    main()