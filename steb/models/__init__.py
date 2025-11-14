from .base import STEBModel
from .hf_model import HFModel
from .luar_model import LUARModel

MODEL_REGISTRY = {
    "hf": HFModel,
    "luar": LUARModel,
}
