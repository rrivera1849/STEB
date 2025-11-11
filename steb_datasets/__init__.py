import os

DATASET_REGISTRY = []
for d in os.listdir("steb_datasets"):
    full_path = os.path.join("steb_datasets", d)
    if os.path.isdir(full_path):
        if os.path.exists(os.path.join(full_path, "config.json")):
            DATASET_REGISTRY.append(d)
DATASET_REGISTRY = sorted(DATASET_REGISTRY)