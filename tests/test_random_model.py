import numpy as np
from transformers import set_seed

from steb.core import get_model
from steb.models.random_model import RandomModel


class TestRandomModelResolution:
    """Verify that the model registry resolves 'random' to RandomModel."""

    def test_get_model_returns_random_model(self):
        model = get_model("random")
        assert isinstance(model, RandomModel)
        assert model.model_name_or_path == "random"


class TestRandomModelEmbedding:
    """Test RandomModel embed_multiple shape and determinism."""

    def test_embed_shape_matches_episodes(self):
        """Output has one row per episode, with the configured embedding dim."""
        model = RandomModel("random")
        episodes = [
            ["The cat sat on the mat."],
            ["Dogs are great pets.", "I love animals."],
            ["Hello there.", "General Kenobi.", "You are a bold one."],
        ]

        result = model.embed_multiple(episodes, batch_size=4)

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(episodes), RandomModel.embedding_dim)
        assert result.dtype == np.float32

    def test_deterministic_under_fixed_seed(self):
        """Re-seeding numpy reproduces identical embeddings."""
        model = RandomModel("random")
        episodes = [["a"], ["b", "c"], ["d"]]

        set_seed(42)
        first = model.embed_multiple(episodes, batch_size=4)

        set_seed(42)
        second = model.embed_multiple(episodes, batch_size=4)

        np.testing.assert_array_equal(first, second)
