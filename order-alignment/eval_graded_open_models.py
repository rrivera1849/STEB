"""
Prompt open Hugging Face instruct models to progressively formalize utterances.

For each input sentence, we run a 5-turn conversation:
1) User: "Directly return the answer. Make the following utterance a bit more formal: <SENTENCE>"
2-5) User: "A bit more formal" (repeated 4 times)

We save one JSON file per model with all generations.
Models are prompted sequentially (one model loaded on GPU at a time).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


SENTENCES: List[str] = [
    "Oh k...i'm watching here:)",
    "K..k:)where are you?how did you performed?",
    "Going for dinner.msg you after.",
    "i see. When we finish we have loads of loans to pay",
    "@user GOD DAMN IT... YOU ARE SO RIGHT. THERE'S NO GOING WRONG WITH THIS STUFF.",
    "@user @user I didn't know Oprah Winfrey knew how to play tennis. .. 😱",
    "@user She is so cute ♡",
]

MODELS: List[str] = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "microsoft/phi-4",
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
]


def _sanitize_model_id(model_id: str) -> str:
    # Safe for filenames across systems
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)


def _ensure_tokenizer_padding(tokenizer) -> None:
    # Some causal LMs don't define pad_token; generation can warn/fail without it.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def _apply_chat_template(
    tokenizer, messages: List[Dict[str, str]]
) -> Dict[str, torch.Tensor]:
    """
    Convert chat messages to model inputs.

    Prefer tokenizer chat template if available; otherwise fall back to a simple
    "User/Assistant" text format.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return {"input_ids": input_ids}
        except Exception:
            # Fall back below
            pass

    # Fallback prompt format
    lines: List[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.title()}: {content}")
    lines.append("Assistant:")
    prompt = "\n".join(lines)
    enc = tokenizer(prompt, return_tensors="pt")
    return enc


@torch.inference_mode()
def _generate_assistant_reply(
    *,
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = _apply_chat_template(tokenizer, messages)
    # Move to model device (works for device_map="auto" with single GPU too).
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Drop None values so generate() doesn't complain for deterministic decoding.
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def progressive_formalization_hf(
    *,
    text: str,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Returns 6 strings: [original, level1, level2, level3, level4, level5]
    """
    results: List[str] = [text]

    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Directly return the answer. Make the following utterance a bit more formal: "
                f"{text}"
            ),
        }
    ]

    # Initial formalization
    assistant = _generate_assistant_reply(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    results.append(assistant)
    messages.append({"role": "assistant", "content": assistant})

    # Four additional iterations
    for _ in range(4):
        messages.append({"role": "user", "content": "A bit more formal"})
        assistant = _generate_assistant_reply(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(assistant)
        messages.append({"role": "assistant", "content": assistant})

    return results


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    _ensure_tokenizer_padding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # If we had to add tokens (e.g., pad token), resize embeddings.
    if getattr(model, "get_input_embeddings", None) is not None:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("order-alignment", "graded_open_model_outputs"),
        help="Directory to write one JSON file per model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 for greedy decoding; >0 enables sampling.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for model_id in MODELS:
        out_path = os.path.join(args.output_dir, f"{_sanitize_model_id(model_id)}.json")
        try:
            print(f"\nLoading model: {model_id}")
            model, tokenizer = load_model_and_tokenizer(model_id)
            print("Model loaded. Generating...")

            per_sentence: List[Dict[str, Any]] = []
            for s in SENTENCES:
                outputs = progressive_formalization_hf(
                    text=s,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                per_sentence.append(
                    {
                        "input": s,
                        "outputs": {
                            "original": outputs[0],
                            "level1": outputs[1],
                            "level2": outputs[2],
                            "level3": outputs[3],
                            "level4": outputs[4],
                            "level5": outputs[5],
                        },
                        "all_levels": outputs,
                    }
                )

            payload: Dict[str, Any] = {
                "model_id": model_id,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                },
                "prompt": {
                    "turn_1": (
                        "Directly return the answer. Make the following utterance a bit more formal: "
                        "SENTENCE"
                    ),
                    "follow_up": "A bit more formal",
                    "n_follow_ups": 4,
                },
                "sentences": per_sentence,
            }
        except Exception as e:
            payload = {
                "model_id": model_id,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "error": repr(e),
            }
        finally:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Wrote: {out_path}")

            # Free GPU memory before the next model.
            if "model" in locals():
                try:
                    del model
                except Exception:
                    pass
            if "tokenizer" in locals():
                try:
                    del tokenizer
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
