"""
TF-IDF-weighted n-gram baseline model.

This model:
- Uses character, token, and POS n-gram TF-IDF features.
  - `norm="l2"`
  - `max_features=2000`
  - `min_df`=1 (keep rare n-grams)
- Fits the TF-IDF vectorizers *unsupervisedly* on the evaluation corpus
  (i.e., the texts passed in via `embed_multiple`), which plays the role
  of "test data" in the STEB pipeline.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple, Union
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import STEBModel


class TFIDFNGModel(STEBModel):
    """
    TF-IDF-based n-gram baseline.

    Embeds each input text as the concatenation of:
      - character n-gram TF-IDF
      - token (word) n-gram TF-IDF
      - POS-tag n-gram TF-IDF

    The vectorizers are fitted *once* per process, the first time
    `embed_multiple` is called, on the full corpus of texts passed in.
    This is deliberately "test-corpus / transductive" and unsupervised
    (uses no labels), mirroring how the other baselines avoid supervised
    training.
    """

    # Name used in `get_model` dispatch (see steb/core.py)
    supported_models = ["tfidfngrams"]

    def __init__(
        self,
        model_name_or_path: str,
        truncate: bool = False,
        max_tokens: Optional[int] = None,
        spacy_model: str = "en_core_web_sm",
        batch_size: int = 200,
        char_ngram_range: Tuple[int, int] = (3, 5),
        tok_ngram_range: Tuple[int, int] = (1, 3), #1,4
        pos_ngram_range: Tuple[int, int] = (1, 3), #1,4
    ) -> None:
        """
        Args:
            model_name_or_path:
                The identifier used by STEB (e.g. `"tfidfngrams"` or
                `"tfidfngrams:/path/to/vectorizers.pkl"`).
            truncate:
                Accepted for API compatibility; ignored (TF-IDF n-grams are
                computed over the full text and do not use a token cap).
            max_tokens:
                Accepted for API compatibility; ignored.
            spacy_model:
                spaCy pipeline for tokenization and POS tagging.
            batch_size:
                Batch size for spaCy's `nlp.pipe`.
            char_ngram_range, tok_ngram_range, pos_ngram_range:
                n-gram ranges for the three TF-IDF vectorizers.
        """
        del truncate, max_tokens
        self.model_name_or_path = model_name_or_path

        self._spacy_model = spacy_model
        self._batch_size = batch_size
        self._char_ngram_range = char_ngram_range
        self._tok_ngram_range = tok_ngram_range
        self._pos_ngram_range = pos_ngram_range

        # Lazy spaCy load
        self._nlp = spacy.load(spacy_model, disable=["ner", "textcat"])
        # Handle long retrieval-style concatenations (episode_size=-1) without E088.
        # Parser/NER are disabled, so raising max_length is safe for this model.
        self._nlp.max_length = max(self._nlp.max_length, 3_000_000)
        # Safety threshold for robust chunked processing even below max_length.
        # Long single-doc calls can trigger native crashes in some spaCy/thinc stacks.
        self._safe_single_doc_chars = 200_000

        # Optional path to pre-fitted vectorizers (when using a global TF-IDF model)
        self._vectorizer_path: str | None = None

        # Vectorizers will be initialized/fitted on first call if not preloaded
        self._char_vec: TfidfVectorizer | None = None
        self._tok_vec: TfidfVectorizer | None = None
        self._pos_vec: TfidfVectorizer | None = None

        # If a serialized vectorizer bundle is provided, load it now.
        if ":" in model_name_or_path:
            _, path = model_name_or_path.split(":", 1)
            if path:
                # Allow both absolute and relative paths.
                self._vectorizer_path = os.path.abspath(path)
                if not os.path.exists(self._vectorizer_path):
                    raise FileNotFoundError(
                        f"TF-IDF n-gram vectorizer file not found: {self._vectorizer_path}"
                    )
                with open(self._vectorizer_path, "rb") as f:
                    bundle = pickle.load(f)
                # Expect a simple dict with three vectorizers.
                self._char_vec = bundle.get("char_vec")
                self._tok_vec = bundle.get("tok_vec")
                self._pos_vec = bundle.get("pos_vec")

    def _text_to_str(self, text: Union[str, List[str]]) -> str:
        """Normalize text (STEB may pass segments as list of strings)."""
        if isinstance(text, list):
            return " ".join(t for t in text if isinstance(t, str))
        return text if isinstance(text, str) else ""

    def _prepare_representations(self, texts: List[str]):
        """
        Given a list of raw strings, return parallel lists:
          - char_texts: raw strings (for char n-grams)
          - tok_texts: whitespace-joined lowercase tokens
          - pos_texts: whitespace-joined coarse POS tags
        """
        if not texts:
            return [], [], []

        # Process with fast path first; if any text is long enough to be risky, or
        # exceeds spaCy's current limit, use robust per-text chunking.
        needs_chunked = any(
            (len(t) > self._safe_single_doc_chars) or (len(t) > self._nlp.max_length)
            for t in texts
        )

        # Fast path: all texts comfortably below single-doc threshold.
        if not needs_chunked:
            docs = list(
                self._nlp.pipe(
                    texts,
                    batch_size=self._batch_size,
                )
            )

            char_texts: List[str] = texts
            tok_texts: List[str] = []
            pos_texts: List[str] = []
            for doc in docs:
                tokens = [t.text.lower() for t in doc]
                pos_tags = [t.pos_ for t in doc]
                tok_texts.append(" ".join(tokens))
                pos_texts.append(" ".join(pos_tags))
            return char_texts, tok_texts, pos_texts

        # Robust path: keep long-doc safety, but still use nlp.pipe whenever possible.
        char_texts: List[str] = texts
        tok_texts: List[str] = [""] * len(texts)
        pos_texts: List[str] = [""] * len(texts)
        chunk_size = min(self._safe_single_doc_chars, max(100_000, self._nlp.max_length - 10_000))

        short_items = [(i, t) for i, t in enumerate(texts) if len(t) <= self._safe_single_doc_chars]
        long_items = [(i, t) for i, t in enumerate(texts) if len(t) > self._safe_single_doc_chars]

        if short_items:
            short_indices = [i for i, _ in short_items]
            short_texts = [t for _, t in short_items]
            for i, doc in zip(
                short_indices,
                self._nlp.pipe(short_texts, batch_size=self._batch_size),
            ):
                tok_texts[i] = " ".join(t.text.lower() for t in doc)
                pos_texts[i] = " ".join(t.pos_ for t in doc)

        for i, text in long_items:
            # Chunk by characters to avoid E088/native-memory blowups on huge inputs.
            # This may split a token at chunk boundaries but is robust for retrieval-style docs.
            chunks = (text[start : start + chunk_size] for start in range(0, len(text), chunk_size))
            token_parts: List[str] = []
            pos_parts: List[str] = []
            for doc_chunk in self._nlp.pipe(chunks, batch_size=self._batch_size):
                token_parts.extend(t.text.lower() for t in doc_chunk)
                pos_parts.extend(t.pos_ for t in doc_chunk)
            tok_texts[i] = " ".join(token_parts)
            pos_texts[i] = " ".join(pos_parts)

        return char_texts, tok_texts, pos_texts

    def _fit_vectorizers(self, char_texts: List[str], tok_texts: List[str], pos_texts: List[str]):
        """
        Fit TF-IDF vectorizers on the provided corpora.

        Settings:
          - analyzer: "char" / "word"
          - norm: "l2"
          - max_features: 2000
          - min_df: 1 (keep rare n-grams)
        """
        self._char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=self._char_ngram_range,
            lowercase=True,
            min_df=1,
            norm="l2",
            max_features=2000,
        )
        self._tok_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=self._tok_ngram_range,
            lowercase=True,
            min_df=1,
            norm="l2",
            max_features=2000,
        )
        self._pos_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=self._pos_ngram_range,
            lowercase=True,
            min_df=1,
            norm="l2",
            max_features=2000,
        )

        # Fit on the full evaluation corpus (unsupervised, test-corpus baseline)
        self._char_vec.fit(char_texts)
        self._tok_vec.fit(tok_texts)
        self._pos_vec.fit(pos_texts)

    def _transform(self, char_texts: List[str], tok_texts: List[str], pos_texts: List[str]) -> np.ndarray:
        """Transform texts with already-fitted vectorizers and concatenate."""
        assert self._char_vec is not None and self._tok_vec is not None and self._pos_vec is not None

        X_char = self._char_vec.transform(char_texts).toarray()
        X_tok = self._tok_vec.transform(tok_texts).toarray()
        X_pos = self._pos_vec.transform(pos_texts).toarray()

        return np.concatenate([X_char, X_tok, X_pos], axis=1).astype(np.float32)

    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of episodes (each a list of texts) as required by STEB.

        The STEB core flattens its internal structure before calling this, so
        here each `episode` is a list of strings corresponding to one
        position-group (e.g., all "most X" snippets for an episode).
        We concatenate each list into a single document string and then
        apply the shared TF-IDF vectorizers.
        """
        if not episodes:
            return np.zeros((0, 0), dtype=np.float32)

        # Normalize and concatenate segments per episode into a single string
        docs: List[str] = []
        for ep in episodes:
            parts = [self._text_to_str(t) for t in ep]
            docs.append(" ".join(parts))

        char_texts, tok_texts, pos_texts = self._prepare_representations(docs)

        # Lazily fit vectorizers on the "test" corpus the first time we're called
        if self._char_vec is None or self._tok_vec is None or self._pos_vec is None:
            self._fit_vectorizers(char_texts, tok_texts, pos_texts)

        X = self._transform(char_texts, tok_texts, pos_texts)
        return X

