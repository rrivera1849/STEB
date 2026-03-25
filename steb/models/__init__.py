from .base import STEBModel
from .causal_model import CausalModel
from .hf_model import HFModel
from .luar_model import LUARModel

MODEL_REGISTRY = {
    "hf": HFModel,
    "causal": CausalModel,
    "luar": LUARModel,
}
