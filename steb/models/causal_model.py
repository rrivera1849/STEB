from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .base import STEBModel, get_model_max_length
from .chunking import chunk_text
from .hf_model import _aggregate_chunks


def last_token_pooling(
    model_output,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts the hidden state of the last non-padding token for each sequence.

    For causal LMs, the last token's representation captures the full
    left-to-right context of the input.

    Args:
        model_output: The output of the model (first element is token embeddings).
        attention_mask: The attention mask indicating real vs. padding tokens.

    Returns:
        A tensor of shape (batch_size, hidden_dim) with per-sequence embeddings.
    """
    token_embeddings = model_output[0]
    # Find the index of the last non-padding token per sequence
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(token_embeddings.shape[0], device=token_embeddings.device)
    return token_embeddings[batch_indices, sequence_lengths]


MAX_CAUSAL_LENGTH = 10240


class CausalModel(STEBModel):
    """
    A Hugging Face causal language model for style text embedding.

    Extracts the last hidden state of the last non-padding token as the
    embedding for each input text. Uses left-padding so that the meaningful
    last token is always at the end of the sequence.
    """
    supported_models = []

    def __init__(self, model_name_or_path: str):
        """
        Initializes the CausalModel.

        Args:
            model_name_or_path: The name or path of the causal language model.
        """
        self.model_name_or_path = model_name_or_path
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embeds a list of episodes, where each episode is a list of texts.

        Uses last-token pooling on the causal LM's hidden states.

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

        max_length = min(get_model_max_length(self.model, self.tokenizer), MAX_CAUSAL_LENGTH)

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
            batch = all_chunks[i:i + batch_size]
            tokenized_batch = self.tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)
            features = self.model(**tokenized_batch)
            # RRS - We can technically just take the last index, since we're doing left-padding
            features = last_token_pooling(features, tokenized_batch["attention_mask"])
            features = features.detach().cpu().float().numpy()
            all_embeddings.append(features)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        all_embeddings = _aggregate_chunks(all_embeddings, chunks_per_text)
        pooled_embeddings = _aggregate_chunks(all_embeddings, lengths)
        assert pooled_embeddings.shape[0] == len(episodes)

        return pooled_embeddings
