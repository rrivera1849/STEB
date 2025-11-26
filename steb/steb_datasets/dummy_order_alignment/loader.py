from typing import Dict, List


def load_dummy_order_alignment_dataset(path: str) -> List[Dict[str, List[str]]]:
    """
    Provides ordered text sequences for two synthetic stylistic axes.

    Each record mimics the `(text, label)` layout used by the other datasets,
    but the `text` field is a list of texts sorted from the most to the least
    intense expression of the style named by `label`.

    For each style label we provide multiple independent ordered sequences
    so that order-alignment can compare several text lists per style.
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

    formality_seq_3 = [
        "Dear Sir or Madam, I hope this correspondence finds you well.",  # Most formal
        "I hope you are doing well today.",
        "Hi, how are you?",
        "hey, what's up?",
        "sup",  # Least formal
    ]

    formality_seq_4 = [
        "It is a pleasure to make your acquaintance.",  # Most formal
        "Nice to meet you.",
        "Good to see you.",
        "hey there!",
        "yo!",  # Least formal
    ]

    formality_seq_5 = [
        "Thank you very much for your thoughtful consideration.",  # Most formal
        "Thank you for your time.",
        "thanks a lot",
        "thanks",
        "thx",  # Least formal
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

    complexity_seq_3 = [
        "The intricately engineered machine operates with remarkable precision under varying environmental conditions.",  # Most complex
        "The carefully designed machine works very precisely.",
        "The machine works well.",
        "The machine runs.",
        "It works.",  # Least complex
    ]

    complexity_seq_4 = [
        "The comprehensive report meticulously analyzes multiple interacting economic indicators over several decades.",  # Most complex
        "The detailed report analyzes many economic indicators over years.",
        "The report looks at economic indicators.",
        "The report looks at the economy.",
        "The report says things about money.",  # Least complex
    ]

    complexity_seq_5 = [
        "The exceptionally elaborate recipe requires numerous precisely timed preparation steps and rare ingredients.",  # Most complex
        "The complicated recipe needs many carefully timed steps.",
        "The recipe has several steps.",
        "The recipe is simple.",
        "You just cook it.",  # Least complex
    ]

    return [
        {"text": formality_seq_1, "label": "formality"},
        {"text": formality_seq_2, "label": "formality"},
        {"text": formality_seq_3, "label": "formality"},
        {"text": formality_seq_4, "label": "formality"},
        {"text": formality_seq_5, "label": "formality"},
        {"text": complexity_seq_1, "label": "complexity"},
        {"text": complexity_seq_2, "label": "complexity"},
        {"text": complexity_seq_3, "label": "complexity"},
        {"text": complexity_seq_4, "label": "complexity"},
        {"text": complexity_seq_5, "label": "complexity"},
    ]

