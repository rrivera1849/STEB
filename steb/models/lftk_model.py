"""
LFTK-based stylometric model for STEB.
Extracts configurable handcrafted linguistic features via LFTK (https://github.com/brucewlee/lftk)
and returns them as embeddings for evaluation.
"""

import json
import os
from typing import List, Optional, Union

import numpy as np
import spacy
import lftk
from tqdm import tqdm

from .base import STEBModel


def _search_feature_keys(**kwargs) -> List[str]:
    """Return list of LFTK feature keys for given search criteria. Handles both list_key and list-of-dicts return."""
    out = lftk.search_features(**kwargs, return_format="list_key")
    if out and isinstance(out[0], dict):
        return [x["key"] for x in out]
    return list(out) if out else []


def _resolve_feature_keys(model_name_or_path: str) -> List[str]:
    """
    Resolve LFTK feature keys from model name or config path.

    - "lftk" -> use default set ("wordsent","typetokenratio","partofspeech","readformula")
    - "lftk:path/to/config.yaml" or "lftk:path/to/config.json" -> load from file

    Config file can contain (all optional, can be combined):
      - feature_keys: list of individual LFTK keys (e.g. ["a_word_ps", "n_noun", "t_word"])
      - families: list of family names -> resolved via lftk.search_features(family=..., ...)
      - domains: list of domain names -> resolved via lftk.search_features(domain=..., ...)
      - language: optional filter for families/domains (e.g. "general")
    """
    if ":" in model_name_or_path:
        _, config_path = model_name_or_path.split(":", 1)
        if not os.path.isabs(config_path):
            # Relative to cwd or STEB root
            if not os.path.exists(config_path):
                alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config_path)
                if os.path.exists(alt):
                    config_path = alt
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                config = json.load(f)
            else:
                import yaml
                config = yaml.safe_load(f)

        keys = list(config.get("feature_keys", []))
        # If language is specified in the config, use it as a filter; otherwise include all languages
        language = config.get("language", None)
        for family in config.get("families", []):
            if language is not None:
                keys.extend(_search_feature_keys(family=family, language=language))
            else:
                keys.extend(_search_feature_keys(family=family))
        for domain in config.get("domains", []):
            if language is not None:
                keys.extend(_search_feature_keys(domain=domain, language=language))
            else:
                keys.extend(_search_feature_keys(domain=domain))
        if keys:
            return list(dict.fromkeys(keys))  # preserve order, remove dupes

        # No explicit keys/families/domains; if language is set, respect it, otherwise return all features
        if language is not None:
            return _search_feature_keys(language=language)
        return _search_feature_keys()

    # Default v1 (Cursor): surface + readability, general only (gede essay detection): 
    # "wordsent", "avgwordsent", "readformula", "typetokenratio": 'v_measure': 0.36447517922184014
    # v2: "wordsent","typetokenratio","partofspeech","readformula": "v_measure": 0.3749137538236487
    default_families = ["wordsent","typetokenratio","partofspeech","readformula"]
    keys = []
    for fam in default_families:
        keys.extend(_search_feature_keys(family=fam, language="general"))
    return list(dict.fromkeys(keys))


class LFTKModel(STEBModel):
    """
    Stylometric model that uses LFTK handcrafted linguistic features as embeddings.
    Feature set is configurable via model_name_or_path (e.g. "lftk" or "lftk:configs/wordsent.yaml").
    """

    supported_models = ["lftk"]

    def __init__(
        self,
        model_name_or_path: str,
        truncate: bool = False,
        max_tokens: Optional[int] = None,
        spacy_model: str = "en_core_web_sm",
        batch_size: int = 200,
    ):
        """
        Args:
            model_name_or_path: "lftk" for default features, or "lftk:path/to/config.yaml" for custom.
            truncate: Accepted for API compatibility; ignored (LFTK reads the
                full text via spaCy and does not use a token cap).
            max_tokens: Accepted for API compatibility; ignored.
            spacy_model: spaCy pipeline for tokenization/parsing (used by LFTK).
            batch_size: Batch size for spaCy nlp.pipe when processing texts.
        """
        del truncate, max_tokens
        self.model_name_or_path = model_name_or_path
        self._feature_keys = _resolve_feature_keys(model_name_or_path)
        self._spacy_model = spacy_model
        self._batch_size = batch_size
        self._n_process = 1
        # Disable parser and NER — surface/POS features only need the tagger.
        # Use the rule-based sentencizer instead of the dependency parser for
        # sentence boundaries, which is much faster.
        self._nlp = spacy.load(spacy_model, disable=["ner", "textcat", "parser"])
        self._nlp.max_length = 10_000_000
        self._nlp.add_pipe("sentencizer")

    def _text_to_str(self, text: Union[str, List[str]]) -> str:
        """Normalize text to a single string (STEB can pass list of segments)."""
        if isinstance(text, list):
            return " ".join(t for t in text if isinstance(t, str))
        return text if isinstance(text, str) else ""

    def _extract_batch(
        self,
        docs: list,
    ) -> List[dict]:
        """Extract LFTK features from a batch of spaCy docs.

        Falls back to per-doc extraction if the batch raises an error
        (e.g. math domain error for empty/short docs).

        Args:
            docs: List of spaCy Doc objects.

        Returns:
            List of feature dicts, one per doc.
        """
        extractor = lftk.Extractor(docs=docs)
        extractor.customize(stop_words=True, punctuations=True, round_decimal=7)
        try:
            feats_list = extractor.extract(features=self._feature_keys)
        except ValueError:
            feats_list = []
            for doc in docs:
                if len(doc) < 2:
                    feats_list.append({k: 0.0 for k in self._feature_keys})
                    continue
                try:
                    single = lftk.Extractor(docs=[doc])
                    single.customize(stop_words=True, punctuations=True, round_decimal=7)
                    feats = single.extract(features=self._feature_keys)
                    feats_list.append(feats if isinstance(feats, dict) else feats[0])
                except (ValueError, ZeroDivisionError):
                    feats_list.append({k: 0.0 for k in self._feature_keys})
        if isinstance(feats_list, dict):
            feats_list = [feats_list]
        return feats_list

    def _extract_features_for_texts(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Extract LFTK feature vectors for a list of text strings.

        Args:
            texts: List of input texts.
            batch_size: Number of texts to process at a time.
            show_progress: Whether to show a progress bar.

        Returns:
            L2-normalized feature array of shape (n_texts, n_features).
        """
        if not texts:
            return np.zeros((0, len(self._feature_keys)), dtype=np.float64)

        normalized = [self._text_to_str(t) for t in texts]

        rows = []
        iterator = range(0, len(normalized), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding (LFTK)", total=len(iterator))

        for i in iterator:
            batch_texts = normalized[i:i + batch_size]
            docs = list(self._nlp.pipe(batch_texts, batch_size=batch_size))
            feats_list = self._extract_batch(docs)

            for doc, feats in zip(docs, feats_list):
                if len(doc) < 2:
                    rows.append([0.0] * len(self._feature_keys))
                else:
                    rows.append([feats.get(k, 0.0) for k in self._feature_keys])

        X = np.array(rows, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        zero_mask = norms <= 1e-10
        nonzero = ~zero_mask
        X_norm = np.where(nonzero, X / np.where(nonzero, norms, 1.0), X)
        return X_norm

    def embed_single(self, texts: List[str], batch_size: int, show_progress: bool = False) -> np.ndarray:
        """Embed a list of single texts; each text becomes one feature vector."""
        return self._extract_features_for_texts(texts, batch_size, show_progress=show_progress)

    def embed_multiple(
        self,
        episodes: List[List[str]],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed a list of episodes.

        Each episode is a list of texts. Returns one vector per episode by
        concatenating the episode's texts into one string and extracting
        LFTK features from that string.

        Args:
            episodes: A list of episodes to embed.
            batch_size: The batch size to use for processing.
            show_progress: Whether to show a progress bar.

        Returns:
            A numpy array of shape (n_episodes, n_features).
        """
        concatenated = []
        for ep in episodes:
            parts = [self._text_to_str(t) for t in ep]
            concatenated.append(" ".join(parts))
        return self._extract_features_for_texts(concatenated, batch_size, show_progress=show_progress)
