import json

import numpy as np
import pytest

import steb.core as steb_core
from steb.core import _is_causal_model, get_model
from steb.models import get_model_registry
from steb.models.sentence_transformer_model import (
    SentenceTransformerModel,
    is_sentence_transformer_model,
)

# Tiny checkpoint from the sentence-transformers test suite (~18MB).
TINY_ST_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

# Adapter-only sentence-transformers repo with no top-level config.json.
ADAPTER_ONLY_REPO = "Blablablab/multilingual-style-representation-Llama-3.2"


class TestLocalDetection:
    """Verify detection on local checkpoint directories."""

    def test_modules_without_config_detected(self, tmp_path):
        (tmp_path / "modules.json").write_text(json.dumps([]))
        assert is_sentence_transformer_model(str(tmp_path)) is True

    def test_modules_with_config_not_detected(self, tmp_path):
        """Repos loadable by AutoModel keep their existing HFModel routing."""
        (tmp_path / "modules.json").write_text(json.dumps([]))
        (tmp_path / "config.json").write_text(json.dumps({}))
        assert is_sentence_transformer_model(str(tmp_path)) is False

    def test_plain_directory_not_detected(self, tmp_path):
        assert is_sentence_transformer_model(str(tmp_path)) is False


class TestHubDetection:
    """Verify detection on Hugging Face Hub repos."""

    def test_adapter_only_repo_detected(self):
        assert is_sentence_transformer_model(ADAPTER_ONLY_REPO) is True

    def test_regular_model_not_detected(self):
        assert is_sentence_transformer_model("gpt2") is False

    def test_nonexistent_repo_not_detected(self):
        assert is_sentence_transformer_model("nonexistent-org/nonexistent-model-xyz") is False


class TestCausalDetectionRobustness:
    """_is_causal_model must not raise on repos without a config.json."""

    def test_adapter_only_repo_returns_false(self):
        assert _is_causal_model(ADAPTER_ONLY_REPO) is False


class TestDispatch:
    """Verify get_model routes sentence-transformers-only checkpoints."""

    def test_st_checkpoint_routes_before_causal(self, monkeypatch):
        """Detection must win over the causal check without loading weights."""
        captured = {}

        class DummySTModel:
            def __init__(self, model_name_or_path, **kwargs):
                captured["name"] = model_name_or_path

        registry = get_model_registry()
        monkeypatch.setitem(registry, "sentence_transformer", DummySTModel)
        monkeypatch.setattr(steb_core, "is_sentence_transformer_model", lambda _: True)

        model = get_model("some/st-only-model")
        assert isinstance(model, DummySTModel)
        assert captured["name"] == "some/st-only-model"


class TestEmbedding:
    """Test SentenceTransformerModel embedding output on a tiny checkpoint."""

    @pytest.fixture(scope="class")
    @classmethod
    def st_model(cls):
        return SentenceTransformerModel(TINY_ST_MODEL)

    def test_embed_shape_matches_episodes(self, st_model):
        episodes = [
            ["The cat sat on the mat."],
            ["Dogs are great pets.", "I love animals."],
            ["Hello there.", "General Kenobi.", "You are a bold one."],
        ]

        result = st_model.embed_multiple(episodes, batch_size=4)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(episodes)
        assert result.shape[1] == st_model.model.get_sentence_embedding_dimension()
        assert not np.isnan(result).any()

    def test_long_text_chunking(self, st_model):
        """Texts beyond the token cap are chunked and pooled, not dropped."""
        long_text = "This is a sentence about writing style. " * 500
        episodes = [[long_text], ["short text"]]

        result = st_model.embed_multiple(episodes, batch_size=4)

        assert result.shape[0] == 2
        assert not np.isnan(result).any()

    def test_long_context_backbone_capped(self, st_model):
        """Absurd native max lengths are capped like CausalModel does."""
        from steb.models.causal_model import MAX_CAUSAL_LENGTH

        assert st_model._resolved_max_length <= MAX_CAUSAL_LENGTH

    def test_truncate_mode(self):
        model = SentenceTransformerModel(TINY_ST_MODEL, truncate=True, max_tokens=64)
        assert model.effective_max_tokens <= 64

        result = model.embed_multiple([["some text " * 100]], batch_size=2)
        assert result.shape[0] == 1
