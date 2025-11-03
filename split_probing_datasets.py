
import os
import sys

import pandas as pd
from tqdm import tqdm

from utils import PROCESSED_DATA_DIR

pd.options.mode.chained_assignment = None

NUM_BINS = 5
TOTAL_DATA = 50_000
TRAIN_P, VAL_P, TEST_P = 0.8, 0.1, 0.1
FILENAME = "reddit.jsonl"

def load_data():
    dataset_name = os.path.join(
        PROCESSED_DATA_DIR, "probing", FILENAME,
    )
    df = pd.read_json(dataset_name, lines=True)
    df = df.sample(frac=1.).reset_index(drop=True)
    return df

def main():
    df = load_data()
    features = [col for col in df.columns if col != "text"]
    for feat in tqdm(features):
        df_feat = df[["text", feat]]

        num_labels = len(pd.qcut(df[feat], q=NUM_BINS, duplicates="drop").value_counts())
        df_feat["label"] = pd.qcut(
            df[feat], q=NUM_BINS, duplicates="drop",
            labels=[str(i) for i in range(num_labels)]
        )

        N_sample = min(df_feat["label"].value_counts().tolist() + [TOTAL_DATA // num_labels])
        df_feat = df_feat.groupby("label").apply(
            lambda x: x.sample(n=N_sample)
        )
        df_feat = df_feat.reset_index(drop=True)
        
        dirname = os.path.join(
            PROCESSED_DATA_DIR, "probing", FILENAME.split(".")[0],
        )
        os.makedirs(dirname, exist_ok=True)
        output_filename = os.path.join(
            dirname, f"{feat}.jsonl",
        )
        if os.path.isfile(output_filename):
            os.remove(output_filename)

        with open(output_filename, "a") as f:
            for i in range(num_labels):
                df_feat_bin = df_feat[df_feat["label"] == str(i)].reset_index(drop=True)
                df_feat_bin["split"] = "train"
                train_size = int(TRAIN_P * len(df_feat_bin))
                val_size = int(VAL_P * len(df_feat_bin))
                df_feat_bin["split"].iloc[:train_size] = "train"
                df_feat_bin["split"].iloc[train_size:train_size+val_size] = "val"
                df_feat_bin["split"].iloc[train_size+val_size:] = "test"
                df_feat_bin.to_json(f, lines=True, orient="records")

    return 0

if __name__ == "__main__":
    sys.exit(main())