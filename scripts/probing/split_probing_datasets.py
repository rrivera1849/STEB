import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = ArgumentParser(
    description="Split a probing feature dataset into a unified JSONL with per-feature labels and train/val/test splits.",
)
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Path to the input JSONL file produced by create_probing_datasets.py.",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to the output unified JSONL file.",
)
parser.add_argument(
    "--num_bins",
    type=int,
    default=5,
    help="Number of quantile bins for labeling.",
)
parser.add_argument(
    "--total_data",
    type=int,
    default=38_400,
    help="Maximum total number of samples across all bins.",
)
parser.add_argument(
    "--train_p",
    type=float,
    default=0.8,
    help="Proportion of data for the train split.",
)
parser.add_argument(
    "--val_p",
    type=float,
    default=0.1,
    help="Proportion of data for the validation split.",
)
args = parser.parse_args()


def load_data(
    input_path: str,
) -> pd.DataFrame:
    """Load and shuffle a probing feature JSONL file.

    Args:
        input_path: Path to the input JSONL file.

    Returns:
        A shuffled DataFrame.
    """
    df = pd.read_json(input_path, lines=True)
    # Shuffling once here handles randomness for the entire pipeline
    df = df.sample(frac=1., random_state=42).reset_index(drop=True)
    return df

def main():
    """Bin each feature into quantile labels, assign splits, and write a unified JSONL."""
    df = load_data(args.input_path)
    features = [col for col in df.columns if col != "text"]

    # Initialize all target columns with standard Python None
    # This ensures Pandas doesn't cast integer labels to floats,
    # and None will automatically serialize to `null` in the final JSONL.
    for feat in features:
        df[f"label_{feat}"] = None
        df[f"split_{feat}"] = None

    for feat in tqdm(features):
        # 1. Execute qcut once and grab the number of bins
        temp_labels, bins = pd.qcut(df[feat], q=args.num_bins, duplicates="drop", retbins=True)
        num_labels = len(bins) - 1

        # Format the categories as strings to mimic your original logic
        temp_labels = pd.qcut(
            df[feat], q=args.num_bins, duplicates="drop",
            labels=[str(i) for i in range(num_labels)]
        )

        # 2. Find the minimum class size to balance the dataset
        label_counts = temp_labels.value_counts()
        N_sample = min(label_counts.tolist() + [args.total_data // num_labels])

        # 3. Downsample and assign splits cleanly
        for label_val in label_counts.index:
            # Extract the absolute indices of the rows belonging to this bin
            bin_indices = df[temp_labels == label_val].index.to_numpy()

            # Randomly select N_sample indices to keep and shuffle them
            kept_indices = np.random.choice(bin_indices, size=N_sample, replace=False)
            np.random.shuffle(kept_indices)

            # Calculate split cutoffs
            train_end = int(args.train_p * N_sample)
            val_end = train_end + int(args.val_p * N_sample)

            # Use .loc to safely assign the label and splits ONLY to the kept rows
            df.loc[kept_indices, f"label_{feat}"] = int(label_val)
            df.loc[kept_indices[:train_end], f"split_{feat}"] = "train"
            df.loc[kept_indices[train_end:val_end], f"split_{feat}"] = "val"
            df.loc[kept_indices[val_end:], f"split_{feat}"] = "test"

    # 4. Create the aggregate tuple/list columns for 'label' and 'split'
    # .values.tolist() is highly optimized for row-wise aggregation
    df["label"] = df[[f"label_{feat}" for feat in features]].values.tolist()
    df["split"] = df[[f"split_{feat}" for feat in features]].values.tolist()

    # 5. Export to a single unified JSONL
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if os.path.isfile(args.output_path):
        os.remove(args.output_path)

    # orient="records" combined with lines=True maps directly to JSONL
    df.to_json(args.output_path, orient="records", lines=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())
