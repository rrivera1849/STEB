"""
Neurobiber baseline: 96-dimensional Biber-style stylistic features via
https://huggingface.co/Blablablab/neurobiber (RoBERTa sequence classification).

Embeddings are sigmoid probabilities per feature, aggregated across 512-token
(word-split) chunks with max pooling, matching the reference inference script.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from .base import STEBModel

DEFAULT_NEUROBIBER_ID = "Blablablab/neurobiber"
CHUNK_WORDS = 512
TOKENIZER_MAX_LENGTH = 512


def chunk_text_by_words(text: str, chunk_size: int = CHUNK_WORDS) -> List[str]:
    """
    Split text into whitespace-token chunks (same convention as the model card).
    Empty or whitespace-only input yields no chunks.
    """
    tokens = text.strip().split()
    if not tokens:
        return []
    return [" ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)]


def _text_to_str(text) -> str:
    if isinstance(text, list):
        return " ".join(t for t in text if isinstance(t, str))
    return text if isinstance(text, str) else ""


class NeurobiberModel(STEBModel):
    """
    Neurobiber multi-label stylistic features as fixed-size vectors for STEB.
    Use ``steb.get_model("neurobiber")`` or ``steb.get_model("neurobiber:ORG/name")``.
    """

    supported_models = ["neurobiber", "Blablablab/neurobiber"]

    def __init__(self, model_name_or_path: str):
        if model_name_or_path.startswith("neurobiber:"):
            _, hf_id = model_name_or_path.split(":", 1)
            hf_id = hf_id.strip() or DEFAULT_NEUROBIBER_ID
        elif model_name_or_path == "neurobiber":
            hf_id = DEFAULT_NEUROBIBER_ID
        else:
            hf_id = model_name_or_path

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name_or_path = hf_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_id)
        self.model.to(self.device)
        self.model.eval()
        self._num_labels = int(self.model.config.num_labels)
        self._chunk_words = CHUNK_WORDS
        self._max_length = TOKENIZER_MAX_LENGTH

    def _predict_proba_for_strings(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        chunked_texts: List[str] = []
        chunk_ranges: List[Tuple[int, int]] = []

        for text in texts:
            start = len(chunked_texts)
            pieces = chunk_text_by_words(text, self._chunk_words)
            chunked_texts.extend(pieces)
            chunk_ranges.append((start, start + len(pieces)))

        n = len(texts)
        dim = self._num_labels
        if not chunked_texts:
            return np.zeros((n, dim), dtype=np.float32)

        import torch

        all_probs: List = []
        n_batches = (len(chunked_texts) + batch_size - 1) // batch_size
        batch_iter = range(0, len(chunked_texts), batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, desc="Neurobiber", total=n_batches)

        for i in batch_iter:
            batch_chunks = chunked_texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_chunks,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self.device)

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = self.model(**enc).logits
                else:
                    logits = self.model(**enc).logits
            all_probs.append(torch.sigmoid(logits).float().cpu())

        flat = torch.cat(all_probs, dim=0)
        out = np.zeros((n, dim), dtype=np.float32)
        for row_idx, (start, end) in enumerate(chunk_ranges):
            if start == end:
                continue
            chunk_preds = flat[start:end]
            out[row_idx] = torch.max(chunk_preds, dim=0).values.numpy().astype(np.float32)
        return out

    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        concatenated = []
        for ep in episodes:
            parts = [_text_to_str(t) for t in ep]
            concatenated.append(" ".join(parts))
        return self._predict_proba_for_strings(concatenated, batch_size, show_progress)
