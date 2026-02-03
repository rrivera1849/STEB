"""
Prompt open Hugging Face instruct models to progressively formalize utterances.
"""
import argparse
import csv
import gc
import os
import re
from datetime import datetime, timezone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from informal_human_data import get_informal_tweet_eval_tweets, get_informal_sms_messages


PROMPT = ("Gradually increase the formality of the following utterance in a small increment. "
          "Whenever I ask for a more formal version, adjust the formality upward only slightly—just one level at a time. "
          "Respond only with the revised sentence and nothing else.")
# PROMPT = "Directly return the answer. Make the following utterance a bit more formal"
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
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen3-VL-8B-Instruct",
    # "google/gemma-3-12b-it",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Ministral-3-14B-Instruct-2512",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    # "HuggingFaceTB/SmolLM2-135M-Instruct",
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # "microsoft/phi-4",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "allenai/OLMo-2-1124-7B-Instruct",
    #"allenai/Olmo-3-7B-Instruct",
    # "allenai/Olmo-3.1-32B-Instruct"
]

N_LEVELS = 4


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try standard loading first, fall back to trust_remote_code model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    except ValueError:
        # For newer Mistral models that need their own model class
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer

@torch.inference_mode()
def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    do_sample = temperature > 0.0
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def formalize_progressively(model, tokenizer, text: str, max_new_tokens: int, temperature: float) -> list[str]:
    """Returns [original, level1, level2, level3, level4]"""
    results = [text]
    messages = [
        {"role": "user", "content": f"{PROMPT}\n\n{text}"}
    ]

    for i in range(N_LEVELS):
        if i > 0:
            messages.append({"role": "user", "content": FOLLOW_UP})

        reply = generate_reply(model, tokenizer, messages, max_new_tokens, temperature)
        results.append(reply)
        messages.append({"role": "assistant", "content": reply})

    return results


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(output_path: str, rows: list[dict]):
    fieldnames = ["model", "source", "input", "level1", "level2", "level3", "level4"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=os.path.join("order-alignment",
                                                                       "model-comparison/graded_open_model_outputs"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-run", action="store_true", help="Run on full dataset instead of test sentences")
    parser.add_argument("--model", type=str, help="Model to run (required for --full-run, ignored otherwise)")
    args = parser.parse_args()

    if args.full_run and not args.model:
        parser.error("--model is required when using --full-run")

    set_seed(args.seed)

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

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    for model_id in models:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)
        output_path = os.path.join(output_dir, f"{safe_name}.csv")

        print(f"\nLoading: {model_id}")
        try:
            model, tokenizer = load_model(model_id)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        rows = []
        try:
            for i, (text, source) in enumerate(sentences):
                print(f"  [{i+1}/{len(sentences)}] {text[:30]}...")
                levels = formalize_progressively(model, tokenizer, text, args.max_new_tokens, args.temperature)
                rows.append({
                    "model": model_id,
                    "source": source,
                    "input": text,
                    "level1": levels[1],
                    "level2": levels[2],
                    "level3": levels[3],
                    "level4": levels[4],
                })
                save_results(output_path, rows)

            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error during generation: {e}")
            save_results(output_path, rows)
        finally:
            unload_model(model, tokenizer)


if __name__ == "__main__":
    main()