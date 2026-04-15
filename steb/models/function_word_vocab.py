"""
Shared function-word list loading and sklearn vocabulary construction.

Used by function-word baselines; vocabulary dicts must use contiguous indices 0..n-1.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

# Bundled list: steb/resources/function_words_phrases.json
FUNCTION_WORDS_JSON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "resources", "function_words_phrases.json")
)
# Backward-compatible alias for tests / older imports
_FUNCTION_WORDS_JSON = FUNCTION_WORDS_JSON


def _assert_sklearn_vocabulary_sane(vocabulary: Dict[str, int]) -> None:
    n = len(vocabulary)
    if n == 0:
        return
    indices = sorted(vocabulary.values())
    if indices != list(range(n)):
        raise ValueError(
            "Vocabulary indices must be exactly 0..n-1 for sklearn vectorizers; "
            f"got {n} terms with indices {indices[:20]}{'...' if len(indices) > 20 else ''}"
        )


def build_function_word_vocabulary(words: List[str], phrases: List[str]) -> Tuple[Dict[str, int], int]:
    """
    Build token -> index with contiguous 0..n-1 indices.

    Lowercasing can collide (e.g. \"I\" and \"i\"); duplicates must be dropped or
    sklearn raises \"Vocabulary of size k doesn't contain index j\".
    """
    vocab_items: List[str] = []
    seen: set[str] = set()
    for w in words:
        t = w.lower()
        if t not in seen:
            seen.add(t)
            vocab_items.append(t)
    for p in phrases:
        t = p.lower()
        if t not in seen:
            seen.add(t)
            vocab_items.append(t)
    vocabulary = {tok: i for i, tok in enumerate(vocab_items)}
    _assert_sklearn_vocabulary_sane(vocabulary)
    max_len = 1
    for phrase in phrases:
        max_len = max(max_len, len(phrase.split()))
    return vocabulary, max_len


# Alias for call sites that used the private name from function_word_model
def _build_vocabulary(words: List[str], phrases: List[str]) -> Tuple[Dict[str, int], int]:
    return build_function_word_vocabulary(words, phrases)
