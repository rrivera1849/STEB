import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .base import STEBModel
from .chunking import chunk_text
from termcolor import colored
from typing import List
from tqdm import tqdm

def get_model_max_length(
    model: AutoModel,
    tokenizer: AutoTokenizer,
) -> int:
    max_len = tokenizer.model_max_length
    if max_len > 100_000:
        max_len = None
    if max_len is None:
        max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None:
        max_len = getattr(model.config, "n_positions", None)
    if max_len is None:
        max_len = 512
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
    attention_mask
):
    """
    Performs mean pooling on the model output.

    Args:
        model_output: The output of the model.
        attention_mask: The attention mask.

    Returns:
        The pooled output.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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

        # Aggregate chunk embeddings back to per-text embeddings
        text_embeddings = []
        start = 0
        for n_chunks in chunks_per_text:
            text_embeddings.append(all_embeddings[start:start+n_chunks].mean(axis=0, keepdims=True))
            start += n_chunks
        all_embeddings = np.concatenate(text_embeddings, axis=0)

        # Pool per-text embeddings within each episode
        pooled_embeddings = []
        start = 0
        for length in lengths:
            pooled_embeddings.append(all_embeddings[start:start+length].mean(axis=0, keepdims=True))
            start += length
        pooled_embeddings = np.concatenate(pooled_embeddings, axis=0)
        assert pooled_embeddings.shape[0] == len(episodes)

        return pooled_embeddings
