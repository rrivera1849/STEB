import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .base import STEBModel
from typing import List
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
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

    def embed_single(self, texts: List[str], batch_size: int, show_progress: bool = False) -> np.ndarray:
        """
        Embeds a list of single texts.

        Args:
            texts: A list of strings to embed.
            batch_size: The batch size to use for embedding.
            show_progress: Whether to show a progress bar.

        Returns:
            A numpy array of embeddings.
        """
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", total=len(iterator))
            
        for i in iterator:
            batch = texts[i:i+batch_size]
            max_length = 512
            tokenized_batch = self.tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized_batch = {
                k: v.to(self.device)
                for k, v in tokenized_batch.items()
            }
            with torch.no_grad():
                features = self.model(**tokenized_batch)
                features = mean_pooling(features, tokenized_batch["attention_mask"])
                features = features.detach().cpu().numpy()
            all_embeddings.append(features)
        return np.concatenate(all_embeddings)

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
        iterator = range(0, len(episodes), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", total=len(iterator))

        for i in iterator:
            batch = episodes[i:i+batch_size]

            # Flatten the batch of episodes into a single list of texts
            texts = [text for episode in batch for text in episode]

            max_length = 512
            tokenized_batch = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized_batch = {
                k: v.to(self.device)
                for k, v in tokenized_batch.items()
            }
            with torch.no_grad():
                features = self.model(**tokenized_batch)
                features = mean_pooling(features, tokenized_batch["attention_mask"])

                # Reshape the features back to the episode structure and average
                episode_size = len(batch[0])
                features = features.reshape(len(batch), episode_size, -1)
                features = features.mean(dim=1)

                features = features.detach().cpu().numpy()
            all_embeddings.append(features)
        return np.concatenate(all_embeddings)
