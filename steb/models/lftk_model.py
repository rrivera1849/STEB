"""
LFTK-based stylometric model for STEB.
Extracts configurable handcrafted linguistic features via LFTK (https://github.com/brucewlee/lftk)
and returns them as embeddings for evaluation.
"""

import json
import os
from typing import List, Union

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

    def __init__(self, model_name_or_path: str, spacy_model: str = "en_core_web_sm", batch_size: int = 200):
        """
        Args:
            model_name_or_path: "lftk" for default features, or "lftk:path/to/config.yaml" for custom.
            spacy_model: spaCy pipeline for tokenization/parsing (used by LFTK).
            batch_size: Batch size for spaCy nlp.pipe when processing texts.
        """
        self.model_name_or_path = model_name_or_path
        self._feature_keys = _resolve_feature_keys(model_name_or_path)
        self._spacy_model = spacy_model
        self._batch_size = batch_size
        # Use multiple processes for spaCy where available for speed
        cpu_count = os.cpu_count() or 1
        self._n_process = max(1, min(4, cpu_count))
        # Disable components not needed for most LFTK features to speed up processing
        self._nlp = spacy.load(spacy_model, disable=["ner", "textcat"]) #NOTE may need to change later

    def _text_to_str(self, text: Union[str, List[str]]) -> str:
        """Normalize text to a single string (STEB can pass list of segments)."""
        if isinstance(text, list):
            return " ".join(t for t in text if isinstance(t, str))
        return text if isinstance(text, str) else ""

    def _extract_features_for_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Extract LFTK feature vectors for a list of text strings. Returns (n_texts, n_features)."""
        if not texts:
            return np.zeros((0, len(self._feature_keys)), dtype=np.float64)

        normalized = [self._text_to_str(t) for t in texts]
        docs = list(
            self._nlp.pipe(
                normalized,
                batch_size=self._batch_size,
                n_process=self._n_process,
            )
        )

        # Single LFTK extractor over the whole batch of docs for speed
        extractor = lftk.Extractor(docs=docs)
        extractor.customize(stop_words=True, punctuations=True, round_decimal=7)  # include stop words and punctuation
        try:
            feats_list = extractor.extract(features=self._feature_keys)
        except ValueError:
            # LFTK can raise math domain error (e.g. log(0)) for empty/short docs in typetokenratio etc.; fall back to per-doc
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
        # LFTK returns a single dict for one doc, list of dicts for multiple docs
        if isinstance(feats_list, dict):
            feats_list = [feats_list]

        rows = []
        # LFTK returns one feature dict per doc; align with docs and keep zeros for very short docs
        for doc, feats in zip(docs, feats_list):
            if len(doc) < 2:
                rows.append([0.0] * len(self._feature_keys))
            else:
                rows.append([feats.get(k, 0.0) for k in self._feature_keys])
        X = np.array(rows, dtype=np.float64)
        # L2-normalize so cosine similarity (used by STEB tasks) is scale-invariant; skip zero vectors to avoid collapsing many points to origin (hurts clustering)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        zero_mask = norms <= 1e-10
        if show_progress:
            # Lightweight debug signal about how many episodes effectively have no signal
            zero_count = int(zero_mask.sum())
            print(f"LFTKModel: {zero_count}/{X.shape[0]} embeddings have zero norm (len(doc) < 2 or all-zero features).")
        nonzero = ~zero_mask
        X_norm = np.where(nonzero, X / np.where(nonzero, norms, 1.0), X)
        return X_norm

    def embed_single(self, texts: List[str], batch_size: int, show_progress: bool = False) -> np.ndarray:
        """Embed a list of single texts; each text becomes one feature vector."""
        return self._extract_features_for_texts(texts, show_progress=show_progress)

    def embed_multiple(
        self, episodes: List[List[str]], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a list of episodes. Each episode is a list of texts (e.g. one per position).
        Returns one vector per episode by concatenating the episode's texts into one string
        and extracting LFTK features from that string.
        """
        # One string per episode: concatenate all texts in the episode
        concatenated = []
        for ep in episodes:
            parts = [self._text_to_str(t) for t in ep]
            concatenated.append(" ".join(parts))
        return self._extract_features_for_texts(concatenated, show_progress=show_progress)
