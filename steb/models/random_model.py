from typing import List

import numpy as np

from .base import STEBModel


class RandomModel(STEBModel):
    """
    A baseline that returns random embeddings for each episode.

    Useful as a chance-level reference point when comparing real embedding
    models on STEB tasks. Determinism is governed by the global numpy
    random state, which ``evaluate()`` seeds via ``transformers.set_seed``
    at the start of each run.
    """
    supported_models = ["random"]
    embedding_dim = 384

    def __init__(
        self,
        model_name_or_path: str,
    ):
        """
        Initializes the RandomModel.

        Args:
            model_name_or_path: The model identifier; expected to be "random".
                Stored so result paths are derived consistently with other models.
        """
        self.model_name_or_path = model_name_or_path

    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Returns random embeddings, one per episode.

        Args:
            episodes: A list of episodes; only its length is used.
            batch_size: Unused. Accepted to match the STEBModel signature.
            show_progress: Unused. Accepted to match the STEBModel signature.

        Returns:
            A numpy array of shape (len(episodes), embedding_dim) sampled
            from a standard normal distribution.
        """
        # Cast to float32 to match the dtype produced by the torch-based models
        # (HFModel, CausalModel, LUARModel all return float32 via .cpu().numpy()).
        return np.random.randn(len(episodes), self.embedding_dim).astype(np.float32)
