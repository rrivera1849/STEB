"""
Function-word frequency baseline model.

Embeds each text with normalized counts over the fixed function-word and
function-phrase vocabulary from `steb/resources/function_words_phrases.json`.

Use: steb.get_model("functionwordfreq")
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .base import STEBModel
from .function_word_vocab import FUNCTION_WORDS_JSON, _build_vocabulary


class FunctionWordFreqModel(STEBModel):
    """
    Count-based baseline using only function words and phrases.

    Features are normalized by document token count (relative frequency),
    giving style-oriented rates rather than corpus-dependent TF-IDF weights.
    """

    supported_models = ["functionwordfreq"]

    def __init__(
        self,
        model_name_or_path: str,
        truncate: bool = False,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Args:
            model_name_or_path: The model identifier; expected to be "functionwordfreq".
            truncate: Accepted for API compatibility; ignored (no tokenizer cap).
            max_tokens: Accepted for API compatibility; ignored.
        """
        del truncate, max_tokens
        self.model_name_or_path = model_name_or_path
        prefix = model_name_or_path.split(":", 1)[0]
        if prefix != "functionwordfreq":
            raise ValueError(
                f"FunctionWordFreqModel expects model name 'functionwordfreq', got {model_name_or_path!r}"
            )
        if not os.path.isfile(FUNCTION_WORDS_JSON):
            raise FileNotFoundError(f"Missing function word list: {FUNCTION_WORDS_JSON}")
        self._vectorizer = self._build_vectorizer(FUNCTION_WORDS_JSON)

    @staticmethod
    def _build_vectorizer(json_path: str) -> CountVectorizer:
        with open(json_path, "r") as f:
            data = json.load(f)

        words = data.get("words", [])
        phrases = data.get("phrases", [])
        vocabulary, max_len = _build_vocabulary(words, phrases)

        # token_pattern keeps single-letter tokens like "a" and "i".
        return CountVectorizer(
            analyzer="word",
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, max_len),
            vocabulary=vocabulary,
        )

    @staticmethod
    def _normalize_texts(episodes: List[List[str]]) -> List[str]:
        docs: List[str] = []
        for ep in episodes:
            parts = [t for t in ep if isinstance(t, str)]
            docs.append(" ".join(parts))
        return docs

    @staticmethod
    def _token_count(text: str) -> int:
        return max(1, len(re.findall(r"(?u)\b\w+\b", text.lower())))

    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not episodes:
            return np.zeros((0, 0), dtype=np.float32)

        docs = self._normalize_texts(episodes)
        X_counts = self._vectorizer.transform(docs).toarray().astype(np.float32)

        # Convert counts to relative frequencies per token.
        lengths = np.array([self._token_count(doc) for doc in docs], dtype=np.float32).reshape(-1, 1)
        return X_counts / lengths

