from models.hf_model import HFModel
from models.luar_model import LUARModel

MODEL_REGISTRY = {
    "hf": HFModel,
    "luar": LUARModel,
}
