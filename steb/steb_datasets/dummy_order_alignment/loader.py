from typing import Dict, List


def load_dummy_order_alignment_dataset(path: str) -> List[Dict[str, List[str]]]:
    """
    Provides ordered text sequences for two synthetic stylistic axes.

    Each record mimics the `(text, label)` layout used by the other datasets,
    but the `text` field is a list of texts sorted from the most to the least
    intense expression of the style named by `label`.

    For each style label we provide at least two independent ordered sequences
    so that order-alignment can compare multiple text lists per style.
    """

    formality_seq_1 = [
        "Greetings, I trust this message finds you in good health.",  # Most formal
        "Good afternoon, I hope you are well.",
        "Hello, how are you today?",
        "hi there, how are you doing?",
        "hey whats up lol",  # Least formal
    ]

    formality_seq_2 = [
        "Salutations, I hope this finds you well.",  # Most formal
        "Good day, I hope you are well.",
        "Hello, how are you?",  # Neutral
        "hey, how's it going?",
        "yo what's going on",  # Least formal
    ]

    complexity_seq_1 = [
        "The exceptionally large dog runs remarkably quickly through the beautifully maintained green park.",  # Most complex
        "The large dog runs quickly through the green park.",
        "The dog runs in the park.",
        "She runs fast.",
        "I run.",  # Least complex
    ]

    complexity_seq_2 = [
        "The incredibly small cat walks extraordinarily carefully across the dangerously busy street.",  # Most complex
        "The small cat walks carefully across the busy street.",
        "The cat walks on the street.",
        "He walks slowly.",
        "We walk.",  # Least complex
    ]

    return [
        {"text": formality_seq_1, "label": "formality"},
        {"text": formality_seq_2, "label": "formality"},
        {"text": complexity_seq_1, "label": "complexity"},
        {"text": complexity_seq_2, "label": "complexity"},
    ]

