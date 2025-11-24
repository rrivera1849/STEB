from typing import Dict, List


def load_dummy_order_alignment_dataset(path: str) -> List[Dict[str, List[str]]]:
    """
    Provides ordered text sequences for two synthetic stylistic axes.

    Each record mimics the `(text, label)` layout used by the other datasets,
    but the `text` field is a list of texts sorted from the most to the least
    intense expression of the style named by `label`.
    """

    formality_descending = [
        "Greetings, I trust this message finds you in good health.",  # Most formal
        "Good afternoon, I hope you are well.",
        "Hello, how are you today?",
        "hi there, how are you doing?",
        "hey whats up lol",  # Least formal
    ]

    complexity_descending = [
        "The exceptionally large dog runs remarkably quickly through the beautifully maintained green park.",  # Most complex
        "The large dog runs quickly through the green park.",
        "The dog runs in the park.",
        "She runs fast.",
        "I run.",  # Least complex
    ]

    return [
        {"text": formality_descending, "label": "formality"},
        {"text": complexity_descending, "label": "complexity"},
    ]

