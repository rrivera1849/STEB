"""LISA (Linguistic Style Analysis) model for style text embedding.

Source: https://ajayp.app/posts/2023/11/learning-interpretable-embeddings-via-llms/
"""
import os
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .base import STEBModel, resolve_token_limit
from .chunking import chunk_text
from transformers import AutoTokenizer

from .lisa_helpers import EncT5ForSequenceClassification
from .hf_model import _aggregate_chunks

LISA_MAX_LENGTH = 512
LISA_NUM_LABELS = 768
LISA_EMBEDDER_FILENAME = "linear_embedder.ckpt"
LISA_BOS_TOKEN_ID = 32100


def is_lisa_model(
    model_name_or_path: str,
) -> bool:
    """Check whether a model path contains a LISA checkpoint.

    Detection is based on the presence of the linear_embedder.ckpt file,
    which is unique to LISA checkpoints.

    Args:
        model_name_or_path: The name or path of the model.

    Returns:
        True if the path contains a LISA checkpoint.
    """
    embedder_path = os.path.join(model_name_or_path, LISA_EMBEDDER_FILENAME)
    return os.path.isfile(embedder_path)


class LISAModel(STEBModel):
    """LISA style embedding model.

    Uses a T5-based encoder to produce style feature vectors, then projects
    them through a linear embedder to produce the final embeddings.
    """

    supported_models = []

    def __init__(
        self,
        model_name_or_path: str,
        truncate: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the LISAModel.

        Args:
            model_name_or_path: Path to the LISA checkpoint directory.
                Must contain pytorch_model.bin, config.json, and linear_embedder.ckpt.
            truncate: If True, truncate each text to the token cap instead
                of chunking and mean-pooling.
            max_tokens: Optional per-text token cap. Capped at LISA's
                native maximum (``LISA_MAX_LENGTH``).
        """
        self.model_name_or_path = model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.truncate = truncate
        self.max_tokens = max_tokens
        self._resolved_max_length = resolve_token_limit(LISA_MAX_LENGTH, max_tokens)
        if truncate or max_tokens is not None:
            self.effective_max_tokens = self._resolved_max_length

        self.model = EncT5ForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=LISA_NUM_LABELS,
            problem_type="regression",
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

        embedder_path = os.path.join(model_name_or_path, LISA_EMBEDDER_FILENAME)
        checkpoint = torch.load(embedder_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        if "embedder" in state_dict:
            self.embedder = state_dict["embedder"]
        else:
            self.embedder = state_dict["embedder.weight"].T
        self.embedder = self.embedder.to(self.device)

    @torch.inference_mode()
    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed a list of episodes, where each episode is a list of texts.

        Args:
            episodes: A list of episodes to embed.
            batch_size: The batch size to use for embedding.
            show_progress: Whether to show a progress bar.

        Returns:
            A numpy array of shape (n_episodes, embedding_dim).
        """
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

        # Encode in batches with adaptive batch size (following LISA's OOM-safe pattern)
        all_predictions = []
        start_idx = 0
        cur_batch_size = batch_size

        if show_progress:
            pbar = tqdm(total=len(all_chunks), desc="Embedding (LISA)")

        while start_idx < len(all_chunks):
            batch = all_chunks[start_idx:start_idx + cur_batch_size]
            # Reserve one token for the BOS we prepend below
            tokenized = self.tokenizer(
                batch,
                truncation=True,
                max_length=self._resolved_max_length - 1,
                padding=True,
                return_tensors="pt",
            )

            # LISA's EncT5Tokenizer expected <s> + tokens + </s>, but the
            # fast T5 tokenizer in transformers >=5 only appends </s>.
            # Manually prepend the BOS token to match the original format.
            bos_ids = torch.full(
                (tokenized["input_ids"].shape[0], 1),
                LISA_BOS_TOKEN_ID,
                dtype=tokenized["input_ids"].dtype,
            )
            bos_mask = torch.ones_like(bos_ids)
            tokenized["input_ids"] = torch.cat([bos_ids, tokenized["input_ids"]], dim=1)
            tokenized["attention_mask"] = torch.cat([bos_mask, tokenized["attention_mask"]], dim=1)
            tokenized = tokenized.to(self.device)

            try:
                predictions = self.model.forward(**tokenized)[0]
            except RuntimeError:
                if cur_batch_size == 1:
                    raise
                cur_batch_size = max(cur_batch_size // 2, 1)
                continue

            all_predictions.append(predictions)
            if show_progress:
                pbar.update(len(batch))
            start_idx += len(batch)
            cur_batch_size = min(cur_batch_size * 2, batch_size)

        if show_progress:
            pbar.close()

        # Clamp to [0, 1] and project through the linear embedder
        raw_vectors = torch.clamp(torch.vstack(all_predictions), min=0.0, max=1.0)
        if len(self.embedder.shape) == 2:
            embeddings = (raw_vectors @ self.embedder).cpu().numpy()
        else:
            embeddings = (raw_vectors * self.embedder).cpu().numpy()

        # Aggregate chunks -> per-text, then per-text -> per-episode
        embeddings = _aggregate_chunks(embeddings, chunks_per_text)
        pooled_embeddings = _aggregate_chunks(embeddings, lengths)
        assert pooled_embeddings.shape[0] == len(episodes)

        return pooled_embeddings
