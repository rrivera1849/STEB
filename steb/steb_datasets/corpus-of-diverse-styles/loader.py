from typing import Any, Dict, Optional, List


def corpus_of_diverse_styles_record_handler(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Custom record handler for the corpus-of-diverse-styles dataset.
    Builds a text list containing both the original text and its paraphrase.
    """
    texts: List[str] = []

    text = example.get("text")
    paraphrase = example.get("paraphrase")

    if isinstance(text, str):
        texts.append(text)
    if isinstance(paraphrase, str):
        texts.append(paraphrase)

    if not texts:
        return None

    return {"text": texts, "label": example.get("label")}

