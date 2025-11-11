
import os
import pandas as pd

def load_hate_speech_dataset(path: str):
    """
        GitHub: https://github.com/Vicomtech/hate-speech-dataset
        Paper: https://aclanthology.org/W18-51.pdf
    """
    records = []

    annotations = pd.read_csv(os.path.join(path, "annotations_metadata.csv"))
    test_dir = os.path.join(path, "sampled_test")
    for fname in os.listdir(test_dir):
        text = open(os.path.join(test_dir, fname), "r").read()
        file_id = os.path.splitext(fname)[0]
        label = annotations[annotations["file_id"] == file_id]["label"].iloc[0]

        records.append({
            "text": text,
            "label": label,
        })

    return records
