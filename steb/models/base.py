from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from termcolor import colored
from transformers import AutoModel, AutoTokenizer

DEFAULT_MAX_LENGTH = 512


def resolve_token_limit(
    model_max: int,
    max_tokens: Optional[int],
) -> int:
    """Compute the effective per-text token cap.

    Args:
        model_max: The model's native maximum sequence length.
        max_tokens: User-specified cap, or None to use the model's max.

    Returns:
        ``min(max_tokens, model_max)`` when ``max_tokens`` is set, else
        ``model_max``.
    """
    if max_tokens is None:
        return model_max
    return min(max_tokens, model_max)


def get_model_max_length(
    model: AutoModel,
    tokenizer: AutoTokenizer,
) -> int:
    """
    Determines the maximum sequence length supported by the model.

    Checks the tokenizer, then model config attributes, falling back to 512.
    Adjusts for RoBERTa-style position embedding offsets.

    Args:
        model: A HuggingFace model.
        tokenizer: The corresponding tokenizer.

    Returns:
        The maximum sequence length the model can handle.
    """
    max_len = tokenizer.model_max_length
    if max_len > 100_000:
        max_len = None
    if max_len is None:
        max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None:
        max_len = getattr(model.config, "n_positions", None)
    if max_len is None:
        max_len = DEFAULT_MAX_LENGTH
        print(colored("Could not determine the maximum length of the model. Defaulting to 512.", "red"))

    # RoBERTa-style models assign position IDs starting at padding_idx+1 rather than 0,
    # so a sequence of length L produces max position ID L+padding_idx. This means the
    # safe max sequence length is max_position_embeddings - padding_idx - 1 (e.g. 512 for
    # RoBERTa where max_position_embeddings=514 and padding_idx=1).
    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if max_position_embeddings is not None:
        try:
            padding_idx = model.embeddings.position_embeddings.padding_idx
            position_padding_idx = padding_idx if padding_idx is not None else 0
        except AttributeError:
            position_padding_idx = 0
        if position_padding_idx > 0:
            max_len = min(max_len, max_position_embeddings - position_padding_idx - 1)

    return max_len


class STEBModel(ABC):
    """
    Abstract base class for text embedding models.

    Subclasses that tokenize input may set ``effective_max_tokens`` in
    their constructor to the resolved per-text token cap when the user
    has opted into truncation mode or specified an explicit ``max_tokens``
    cap. ``None`` means default chunk-and-pool behavior with no tagging.
    """

    effective_max_tokens: Optional[int] = None

    @abstractmethod
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
        pass
