import os

DATASET_REGISTRY = [
    d for d in os.listdir("steb_datasets") if os.path.isdir(os.path.join("steb_datasets", d))
]
