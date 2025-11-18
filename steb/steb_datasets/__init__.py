import os

DATASET_REGISTRY = [
    f.name
    for f in os.scandir(os.path.dirname(__file__))
    if f.is_dir() and os.path.exists(os.path.join(f.path, "config.json"))
]