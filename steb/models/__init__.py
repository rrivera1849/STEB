from .base import STEBModel
from .causal_model import CausalModel
from .hf_model import HFModel
from .lisa_model import LISAModel
from .luar_model import LUARModel
from .random_model import RandomModel

MODEL_REGISTRY = {
    "hf": HFModel,
    "causal": CausalModel,
    "lisa": LISAModel,
    "luar": LUARModel,
    "random": RandomModel,
}
