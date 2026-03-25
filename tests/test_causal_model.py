import numpy as np
import pytest

from steb.core import get_model, _is_causal_model
from steb.models.causal_model import CausalModel, last_token_pooling


CAUSAL_MODEL_NAME = "gpt2"


@pytest.fixture(scope="module")
def causal_model():
    """Load GPT-2 once for this module."""
    return CausalModel(CAUSAL_MODEL_NAME)


class TestAutoDetection:
    """Verify that get_model auto-detects causal vs encoder models."""

    def test_gpt2_detected_as_causal(self):
        assert _is_causal_model("gpt2") is True

    def test_bert_detected_as_encoder(self):
        assert _is_causal_model("bert-base-uncased") is False

    def test_get_model_returns_causal_for_gpt2(self):
        model = get_model("gpt2")
        assert isinstance(model, CausalModel)


class TestCausalModel:
    """Test CausalModel embedding extraction with GPT-2."""

    def test_embed_single_episode(self, causal_model):
        """Single episode with one text produces a valid embedding."""
        episodes = [["Hello, world!"]]
        result = causal_model.embed_multiple(episodes, batch_size=4)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.shape[1] == causal_model.model.config.hidden_size

    def test_embed_multiple_episodes(self, causal_model):
        """Multiple episodes produce correctly shaped embeddings."""
        episodes = [
            ["The cat sat on the mat."],
            ["Dogs are great pets.", "I love animals."],
            ["Hello there."],
        ]
        result = causal_model.embed_multiple(episodes, batch_size=4)

        assert result.shape[0] == len(episodes)
        assert result.shape[1] == causal_model.model.config.hidden_size

    def test_different_texts_produce_different_embeddings(self, causal_model):
        """Semantically different texts should yield distinct embeddings."""
        episodes = [
            ["The quick brown fox jumps over the lazy dog."],
            ["Quantum physics describes the behavior of particles at atomic scales."],
        ]
        result = causal_model.embed_multiple(episodes, batch_size=4)

        similarity = np.dot(result[0], result[1]) / (
            np.linalg.norm(result[0]) * np.linalg.norm(result[1])
        )
        assert similarity < 0.99, "Unrelated texts should not have near-identical embeddings"

    def test_identical_texts_produce_identical_embeddings(self, causal_model):
        """Identical texts should produce the same embedding."""
        text = "This is a test sentence."
        episodes = [[text], [text]]
        result = causal_model.embed_multiple(episodes, batch_size=4)

        np.testing.assert_array_almost_equal(result[0], result[1])

    def test_left_padding_configured(self, causal_model):
        """Causal models should use left padding."""
        assert causal_model.tokenizer.padding_side == "left"
        assert causal_model.tokenizer.pad_token is not None


class TestLastTokenPooling:
    """Unit tests for the last_token_pooling function."""

    def test_basic_pooling(self):
        """Extracts the correct last-token embedding given an attention mask."""
        import torch

        # batch_size=2, seq_len=4, hidden_dim=3
        hidden_states = torch.tensor([
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # last real at idx 2
            [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [7.0, 0.0, 0.0]],  # last real at idx 3
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 0],  # 3 real tokens
            [1, 1, 1, 1],  # 4 real tokens
        ])

        class FakeOutput:
            def __init__(self, h):
                self._h = h

            def __getitem__(self, idx):
                return self._h

        result = last_token_pooling(FakeOutput(hidden_states), attention_mask)

        assert result.shape == (2, 3)
        torch.testing.assert_close(result[0], torch.tensor([3.0, 0.0, 0.0]))
        torch.testing.assert_close(result[1], torch.tensor([7.0, 0.0, 0.0]))
