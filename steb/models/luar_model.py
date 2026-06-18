from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .base import STEBModel, resolve_token_limit
from .chunking import chunk_text


class LUARModel(STEBModel):
    """
    LUAR (Learning Universal Authorship Representations) models.
    """
    supported_models = ["rrivera1849/LUAR-CRUD", "rrivera1849/LUAR-MUD"]

    def __init__(
        self,
        model_name_or_path: str,
        truncate: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """
        Initializes the LUARModel.

        Args:
            model_name_or_path: The name or path of the LUAR model.
            truncate: If True, truncate each text to the token cap instead
                of chunking. Default False preserves chunk expansion.
            max_tokens: Optional per-text token cap. Capped at the model's
                native maximum.
        """
        self.model_name_or_path = model_name_or_path
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.truncate = truncate
        self.max_tokens = max_tokens
        self._resolved_max_length = resolve_token_limit(self.tokenizer.model_max_length, max_tokens)
        if truncate or max_tokens is not None:
            self.effective_max_tokens = self._resolved_max_length

    @torch.inference_mode()
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
        # Expands out texts that are longer than the model's max length, so the episodes
        # could be of varying lengths. In truncate mode, texts are kept 1:1 and the
        # tokenizer truncates below.
        max_length = self._resolved_max_length
        if not self.truncate:
            episodes = [
                [chunk for text in episode for chunk in chunk_text(text, self.tokenizer, max_length)]
                for episode in episodes
            ]

        all_embeddings = [None] * len(episodes)
        lengths = [len(x) for x in episodes]
        unique_lengths = np.unique(lengths)

        if show_progress:
            pbar = tqdm(total=len(episodes), desc="Embedding")

        for length in unique_lengths:
            indices_to_embed = [i for i, l in enumerate(lengths) if l == length]

            for batch_start in range(0, len(indices_to_embed), batch_size):
                batch_indices = indices_to_embed[batch_start:batch_start+batch_size]
                batch = [episodes[i] for i in batch_indices]
                texts = [text for episode in batch for text in episode]

                tokenized_batch = self.tokenizer(
                    texts,
                    max_length=max_length,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                ).to(self.device)
                longest_length = tokenized_batch["input_ids"].size(1)

                tokenized_batch["input_ids"] = \
                    tokenized_batch["input_ids"].reshape(len(batch), length, longest_length)
                tokenized_batch["attention_mask"] = \
                    tokenized_batch["attention_mask"].reshape(len(batch), length, longest_length)

                features = self.model(
                    input_ids=tokenized_batch["input_ids"],
                    attention_mask=tokenized_batch["attention_mask"]
                ).detach().cpu().numpy()

                for i, idx in enumerate(batch_indices):
                    all_embeddings[idx] = features[i:i+1]

                if show_progress:
                    pbar.update(len(batch_indices))

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings