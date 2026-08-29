import os
from typing import List, Optional

import numpy as np
from huggingface_hub import file_exists

from .base import DEFAULT_MAX_LENGTH, STEBModel, resolve_token_limit
from .chunking import chunk_text

ST_MODULES_FILENAME = "modules.json"
HF_CONFIG_FILENAME = "config.json"


def is_sentence_transformer_model(
    model_name_or_path: str,
) -> bool:
    """Check whether a model is a sentence-transformers-only checkpoint.

    Detection is based on the presence of ``modules.json`` (unique to the
    sentence-transformers format) combined with the absence of a top-level
    ``config.json``. Repos that have both (e.g. all-MiniLM) remain loadable
    by ``AutoModel`` and keep their existing HFModel routing; adapter-only
    repos (e.g. PEFT adapters saved via sentence-transformers) have no
    ``config.json`` and can only be loaded through this class.

    Args:
        model_name_or_path: The name or path of the model.

    Returns:
        True if the model is a sentence-transformers-only checkpoint.
    """
    if os.path.isdir(model_name_or_path):
        has_modules = os.path.isfile(os.path.join(model_name_or_path, ST_MODULES_FILENAME))
        has_config = os.path.isfile(os.path.join(model_name_or_path, HF_CONFIG_FILENAME))
        return has_modules and not has_config

    try:
        has_modules = file_exists(model_name_or_path, ST_MODULES_FILENAME)
        has_config = file_exists(model_name_or_path, HF_CONFIG_FILENAME)
    except Exception:
        return False
    return has_modules and not has_config


class SentenceTransformerModel(STEBModel):
    """
    Wraps a sentence-transformers checkpoint for style text embedding.

    Handles the modular sentence-transformers format (modules.json, pooling
    config, PEFT adapters) that ``AutoModel`` cannot load directly.
    """
    supported_models = []

    def __init__(
        self,
        model_name_or_path: str,
        truncate: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """
        Initializes the SentenceTransformerModel.

        Args:
            model_name_or_path: The name or path of the checkpoint.
            truncate: If True, truncate each text to the token cap instead
                of chunking and mean-pooling.
            max_tokens: Optional per-text token cap. Capped at the model's
                native maximum. ``None`` means use the model's native max.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name_or_path = model_name_or_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on GPU keeps SDPA on the flash/memory-efficient kernels; fp32
        # with an attention mask falls back to the math backend, which
        # materializes the full attention matrix and OOMs on long chunks.
        model_kwargs = {}
        if device == "cuda" and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer

        from .causal_model import MAX_CAUSAL_LENGTH

        self.truncate = truncate
        self.max_tokens = max_tokens
        # Long-context backbones (e.g. Llama 3.2 reports 131072) would make
        # chunk_text a no-op and OOM at encode time; apply the same cap that
        # CausalModel uses for auto-regressive backbones.
        model_max = self.model.get_max_seq_length() or DEFAULT_MAX_LENGTH
        model_max = min(model_max, MAX_CAUSAL_LENGTH)
        self._resolved_max_length = resolve_token_limit(model_max, max_tokens)
        self.model.max_seq_length = self._resolved_max_length
        if truncate or max_tokens is not None:
            self.effective_max_tokens = self._resolved_max_length

    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embeds a list of episodes, where each episode is a list of texts.

        Args:
            episodes: A list of episodes to embed.
            batch_size: The batch size to use for embedding.
            show_progress: Whether to show a progress bar.

        Returns:
            A numpy array of embeddings.
        """
        from .hf_model import _aggregate_chunks

        lengths = [len(x) for x in episodes]
        texts = [text for episode in episodes for text in episode]

        if self.truncate:
            all_chunks = list(texts)
            chunks_per_text = [1] * len(texts)
        else:
            all_chunks = []
            chunks_per_text = []
            for text in texts:
                chunks = chunk_text(text, self.tokenizer, self._resolved_max_length)
                all_chunks.extend(chunks)
                chunks_per_text.append(len(chunks))

        all_embeddings = self.model.encode(
            all_chunks,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Aggregate chunk embeddings -> per-text, then per-text -> per-episode
        all_embeddings = _aggregate_chunks(all_embeddings, chunks_per_text)
        pooled_embeddings = _aggregate_chunks(all_embeddings, lengths)
        assert pooled_embeddings.shape[0] == len(episodes)

        return pooled_embeddings
