import os

DATASET_REGISTRY = [
    d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))
]
