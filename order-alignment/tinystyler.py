"""
Code to use TinyStyler for interpolation, with original code copied from https://huggingface.co/tinystyler/tinystyler
"""

import torch
import importlib
from huggingface_hub import hf_hub_download
from transformers import set_seed
from typing import List, Tuple

# import pydevd_pycharm
# pydevd_pycharm.settrace('hpcs05', port=5678, stdout_to_server=True, stderr_to_server=True)

def load_tinystyler_model(device: str):
    """Load the TinyStyler model and tokenizer from HuggingFace."""
    tinystyler_module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(
            "tinystyler",
            hf_hub_download(repo_id="tinystyler/tinystyler", filename="tinystyler.py"),
        )
    )
    tinystyler_module.__spec__.loader.exec_module(tinystyler_module)

    get_tinystyler_model = tinystyler_module.get_tinystyler_model
    get_target_style_embeddings = tinystyler_module.get_target_style_embeddings

    tokenizer, model = get_tinystyler_model(device)

    return tokenizer, model, get_target_style_embeddings


def calculate_interpolated_style_embeddings(
    source_texts: List[str],
    target_texts: List[str],
    interpolation_factor: float,
    get_style_embeddings_fn,
    device: str
) -> torch.Tensor:
    """
    Calculate interpolated style embeddings between source and target styles.

    Args:
        source_texts: List of example texts representing the source style
        target_texts: List of example texts representing the target style
        interpolation_factor: Value between 0.0 (pure source) and 1.0 (pure target)
        get_style_embeddings_fn: Function to calculate style embeddings
        device: Device to use for computations

    Returns:
        Interpolated style embeddings tensor
    """
    source_embeddings = get_style_embeddings_fn([source_texts], device).to(device)
    target_embeddings = get_style_embeddings_fn([target_texts], device).to(device)

    # Linear interpolation: (1 - a) * source + a * target
    interpolated = (1 - interpolation_factor) * source_embeddings + interpolation_factor * target_embeddings

    return interpolated


def generate_text_with_style(
    source_text: str,
    style_embeddings: torch.Tensor,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> str:
    """
    Generate text with the given style embeddings.

    Args:
        source_text: The input text to transform
        style_embeddings: The style embeddings to apply
        tokenizer: The tokenizer
        model: The TinyStyler model
        device: Device to use
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Generated text string
    """
    inputs = tokenizer(
        [source_text], padding="longest", truncation=True, return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        style=style_embeddings,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return generated_text


def main():
    """Main function to demonstrate style interpolation."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    print(f"Using device: {device}\n")
    print("Loading TinyStyler model...")
    tokenizer, model, get_style_embeddings = load_tinystyler_model(device)
    print("Model loaded successfully!\n")

    # Define inputs
    source_text = "I would like to see the football game."
    source_style_texts = [source_text]
    # Examples of source style (formal):
    # source_style_texts = [
    #     # "I would like to attend the meeting.",
    #     # "Please submit the report by Friday.",
    #     # "We appreciate your cooperation."
    #     "He has a very distinct walk.",
    #     "Santa was excessively overweight while the woman was driving.",
    #     "They cannot see anything in the beginning.",
    #     "Take a look at yourself in the mirror and laugh.",
    #     "And the indication is: At the moment, the Hip Hop genre is decreasing in quality!"
    # ]

    # Examples of target style (informal):
    # target_style_texts = [
    #     "idk.....but i have faith in you lol",
    #     "cant wait for a new album from him.",
    #     "i can't believe it!!1",
    #     "But has a lil slang 2 his walk.",
    #     "santa was to fat, and the woman was driving.",
    #     "Well, they probably can't see anything at first.",
    #     "LOOK AT UR FACE IN THE MIRROR ************LOL****",
    #     "And it says: Right now, Hip Hop music is going DOWNHILL!!!!!!!!!!!"
    # ]

    # source_style_texts = [
    #     "I was tired, so I went to bed early.",
    #     "The student did not understand the lesson.",
    #     "It was raining, so we stayed inside.",
    #     "I think this movie is good.",
    #     "He missed the bus and arrived late."
    # ]

    target_style_texts = [
        "Due to a significant sense of physical and mental fatigue, I made the deliberate decision to retire for the evening at an earlier-than-usual hour.",
        "The learner experienced difficulty comprehending the instructional material presented during the lesson.",
        "Because persistent rainfall created unfavorable outdoor conditions, we chose to remain indoors for the duration of the period.",
        "In my assessment, the film demonstrates a high level of quality and is deserving of positive evaluation.",
        "After failing to catch the scheduled bus, he consequently arrived at his destination later than intended."
    ]

    print(f"Source text: '{source_text}'")
    print(f"\nSource style examples (formal): {source_style_texts}")
    print(f"Target style examples (informal): {target_style_texts}")
    print("\n" + "="*80 + "\n")
    print(f"[0.0] {"ORIGINAL":20s} → '{source_text}'")

    # Generate outputs for interpolation factors from 0.0 to 1.0
    interpolation_values = [i / 5 for i in range(1,6)]  # [0.0, 0.1, 0.2, ..., 1.0]

    for factor in interpolation_values:
        style_embeddings = calculate_interpolated_style_embeddings(
            source_style_texts,
            target_style_texts,
            factor,
            get_style_embeddings,
            device
        )

        generated_text = generate_text_with_style(
            source_text,
            style_embeddings,
            tokenizer,
            model,
            device
        )

        # Format the output nicely
        style_label = "INTERPOLATED"
        print(f"[{factor:.1f}] {style_label:20s} → '{generated_text}'")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()