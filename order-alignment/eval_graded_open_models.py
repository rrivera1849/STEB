"""
Prompt open Hugging Face instruct models to progressively formalize utterances.
"""
import gc
import json
import os
import re
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


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
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "microsoft/phi-4",
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
]

N_LEVELS = 4
OUTPUT_DIR = os.path.join("order-alignment", "graded_open_model_outputs")


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int = 96) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def formalize_progressively(model, tokenizer, text: str) -> list[str]:
    """Returns [original, level1, level2, level3, level4]"""
    results = [text]
    messages = [
        {"role": "user", "content": f"Directly return the answer. Make the following utterance a bit more formal: {text}"}
    ]

    for i in range(N_LEVELS):
        if i > 0:
            messages.append({"role": "user", "content": "More formal"})

        reply = generate_reply(model, tokenizer, messages)
        results.append(reply)
        messages.append({"role": "assistant", "content": reply})

    return results


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(output_path: str, model_id: str, results: list):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_id, "results": results}, f, ensure_ascii=False, indent=2)


def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    for model_id in MODELS:
        print(f"\nLoading: {model_id}")
        model, tokenizer = load_model(model_id)

        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        results = []

        for i, sentence in enumerate(SENTENCES):
            print(f"  [{i+1}/{len(SENTENCES)}] {sentence[:30]}...")
            levels = formalize_progressively(model, tokenizer, sentence)
            results.append({"input": sentence, "levels": levels})
            save_results(output_path, model_id, results)

        print(f"  Saved: {output_path}")
        unload_model(model, tokenizer)


if __name__ == "__main__":
    main()