"""
Prompt OpenAI chat models to progressively formalize utterances.

For each input sentence, we run a 4-turn conversation:
1) User: "Directly return the answer. Make the following utterance a bit more formal: <SENTENCE>"
2-4) User: "A bit more formal" (repeated 3 times)

We save one JSON file per model with all generations.
"""
import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _utc_now_iso_z() -> str:
    # timezone-aware UTC timestamp, formatted like 2026-01-16T12:34:56.789Z
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _utc_run_id() -> str:
    # Compact, filesystem-friendly run id, e.g. 20260116_123456Z
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

TURN_1_PREFIX = "Directly return the answer. Make the following utterance a bit more formal: "
FOLLOW_UP_TEXT = "A bit more formal"
N_FOLLOW_UPS = 3
SENTENCE_PLACEHOLDER = "SENTENCE"


SENTENCES: List[str] = [
    "Oh k...i'm watching here:)"# ,
    # "K..k:)where are you?how did you performed?",
    # "Going for dinner.msg you after.",
    # "i see. When we finish we have loads of loans to pay",
    # "@user GOD DAMN IT... YOU ARE SO RIGHT. THERE'S NO GOING WRONG WITH THIS STUFF.",
    # "@user @user I didn't know Oprah Winfrey knew how to play tennis. .. 😱",
    # "@user She is so cute ♡",
]


MODELS: List[str] = [
    # "gpt-5.1-chat-latest",
    "gpt-5-nano-2025-08-07",
    # "gpt-5-mini-2025-08-07",
]


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)


def _chat_generate(
    *,
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    """
    Minimal wrapper around OpenAI chat completions.
    Tries newer/older token-limit parameter names for compatibility.
    """
    base_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }

    try:
        resp = client.chat.completions.create(
            **base_kwargs,
            max_completion_tokens=max_new_tokens,
        )
    except TypeError:
        resp = client.chat.completions.create(
            **base_kwargs,
            max_tokens=max_new_tokens,
        )

    content = resp.choices[0].message.content
    return (content or "").strip()


def progressive_formalization_openai(
    *,
    text: str,
    client: Any,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> List[str]:
    """
    Returns 5 strings: [original, level1, level2, level3, level4]
    """
    results: List[str] = [text]

    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": f"{TURN_1_PREFIX}{text}",
        }
    ]

    # Initial formalization
    assistant = _chat_generate(
        client=client,
        model=model,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    results.append(assistant)
    messages.append({"role": "assistant", "content": assistant})

    # Three additional iterations
    for _ in range(N_FOLLOW_UPS):
        messages.append({"role": "user", "content": FOLLOW_UP_TEXT})
        assistant = _chat_generate(
            client=client,
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        results.append(assistant)
        messages.append({"role": "assistant", "content": assistant})

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("order-alignment", "graded_closed_models"),
        help="Base directory for outputs (a timestamped subfolder is created per run).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="0.0 for greedy decoding; >0 enables sampling.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = os.path.join(args.output_dir, _utc_run_id())
    os.makedirs(run_dir, exist_ok=True)

    if OpenAI is None:
        raise ImportError(
            "Missing dependency 'openai'. Install it (e.g. `pip install openai`) "
            "and ensure OPENAI_API_KEY is set."
        )

    client = OpenAI()

    for model_id in MODELS:
        out_path = os.path.join(run_dir, f"{_sanitize_model_id(model_id)}.json")
        try:
            per_sentence: List[Dict[str, Any]] = []
            for s in SENTENCES:
                outputs = progressive_formalization_openai(
                    text=s,
                    client=client,
                    model=model_id,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
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
                        },
                        "all_levels": outputs,
                    }
                )

            payload: Dict[str, Any] = {
                "model_id": model_id,
                "generated_at": _utc_now_iso_z(),
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                },
                "prompt": {
                    "turn_1": f"{TURN_1_PREFIX}{SENTENCE_PLACEHOLDER}",
                    "follow_up": FOLLOW_UP_TEXT,
                    "n_follow_ups": N_FOLLOW_UPS,
                },
                "sentences": per_sentence,
            }
        except Exception as e:
            payload = {
                "model_id": model_id,
                "generated_at": _utc_now_iso_z(),
                "error": repr(e),
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

