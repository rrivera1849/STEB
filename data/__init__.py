import os

DATASET_REGISTRY = [
    d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))
]
