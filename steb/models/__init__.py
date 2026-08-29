from __future__ import annotations

from typing import Any, Dict, Optional

from .base import STEBModel

# Built lazily so importing e.g. ``steb.models.neurobiber_model`` does not load
# torch / spaCy / LFTK until ``get_model_registry()`` runs.
_registry: Optional[Dict[str, Any]] = None


def get_model_registry() -> Dict[str, Any]:
    global _registry
    if _registry is None:
        from .causal_model import CausalModel
        from .function_word_freq_model import FunctionWordFreqModel
        from .hf_model import HFModel
        from .lftk_model import LFTKModel
        from .lisa_model import LISAModel
        from .luar_model import LUARModel
        from .neurobiber_model import NeurobiberModel
        from .random_model import RandomModel
        from .sentence_transformer_model import SentenceTransformerModel
        from .tfidf_ngram_model import TFIDFNGModel

        _registry = {
            "hf": HFModel,
            "causal": CausalModel,
            "lisa": LISAModel,
            "sentence_transformer": SentenceTransformerModel,
            "luar": LUARModel,
            "random": RandomModel,
            "lftk": LFTKModel,
            "tfidfngrams": TFIDFNGModel,
            "functionwordfreq": FunctionWordFreqModel,
            "neurobiber": NeurobiberModel,
        }
    return _registry
