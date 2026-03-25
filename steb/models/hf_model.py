from typing import List

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .base import STEBModel
from .chunking import chunk_text

DEFAULT_MAX_LENGTH = 512


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


def mean_pooling(
    model_output,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Averages token embeddings weighted by the attention mask.

    Args:
        model_output: The output of the model (first element is token embeddings).
        attention_mask: The attention mask indicating real vs. padding tokens.

    Returns:
        A tensor of shape (batch_size, hidden_dim) with pooled embeddings.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def _aggregate_chunks(
    embeddings: np.ndarray,
    chunk_counts: List[int],
) -> np.ndarray:
    """
    Mean-pools contiguous slices of embeddings according to chunk_counts.

    Args:
        embeddings: Array of shape (total_chunks, hidden_dim).
        chunk_counts: Number of chunks per group. Must sum to embeddings.shape[0].

    Returns:
        Array of shape (len(chunk_counts), hidden_dim) with one embedding per group.
    """
    pooled = []
    start = 0
    for count in chunk_counts:
        pooled.append(embeddings[start:start + count].mean(axis=0, keepdims=True))
        start += count
    return np.concatenate(pooled, axis=0)


class HFModel(STEBModel):
    """
    A generic Hugging Face model for style text embedding.
    This class serves as a fallback for any model that is not explicitly supported.
    """
    supported_models = []

    def __init__(self, model_name_or_path: str):
        """
        Initializes the HFModel.

        Args:
            model_name_or_path: The name or path of the Hugging Face model.
        """
        self.model_name_or_path = model_name_or_path
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_multiple(self, episodes: List[List[str]], batch_size: int, show_progress: bool = False) -> np.ndarray:
        """
        Embeds a list of episodes, where each episode is a list of texts.

        Args:
            episodes: A list of episodes to embed.
            batch_size: The batch size to use for embedding.
            show_progress: Whether to show a progress bar.

        Returns:
            A numpy array of embeddings.
        """
        all_embeddings = []
        lengths = [len(x) for x in episodes]
        texts = [text for episode in episodes for text in episode]

        max_length = get_model_max_length(self.model, self.tokenizer)

        # Chunk texts that exceed the model's context length
        all_chunks = []
        chunks_per_text = []
        for text in texts:
            chunks = chunk_text(text, self.tokenizer, max_length)
            all_chunks.extend(chunks)
            chunks_per_text.append(len(chunks))

        iterator = range(0, len(all_chunks), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", total=len(iterator))

        for i in iterator:
            batch = all_chunks[i:i+batch_size]
            tokenized_batch = self.tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)
            features = self.model(**tokenized_batch)
            features = mean_pooling(features, tokenized_batch["attention_mask"])
            features = features.detach().cpu().numpy()
            all_embeddings.append(features)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Aggregate chunk embeddings -> per-text, then per-text -> per-episode
        all_embeddings = _aggregate_chunks(all_embeddings, chunks_per_text)
        pooled_embeddings = _aggregate_chunks(all_embeddings, lengths)
        assert pooled_embeddings.shape[0] == len(episodes)

        return pooled_embeddings
