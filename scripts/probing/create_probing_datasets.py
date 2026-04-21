
import os
import random
import sys
from argparse import ArgumentParser
from itertools import compress
from multiprocessing import Pool
from typing import List

import lftk
import pandas as pd
import spacy
from termcolor import colored
from tqdm import tqdm

parser = ArgumentParser(
    description="Extract foundation probing features from a text dataset.",
)

source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument(
    "--input_path",
    type=str,
    help="Path to a JSONL file containing the input text data.",
)
source_group.add_argument(
    "--hf_dataset",
    type=str,
    help="HuggingFace dataset ID (e.g. 'imdb', 'amazon_polarity').",
)

parser.add_argument(
    "--hf_split",
    type=str,
    default="train",
    help="HuggingFace dataset split to load (default: 'train'). Only used with --hf_dataset.",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to the output JSONL file with extracted features.",
)
parser.add_argument(
    "--text_column",
    type=str,
    default="text",
    help="Name of the column containing text.",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=40_000,
    help="Number of documents to keep in the final output.",
)
parser.add_argument(
    "--num_docs_per_worker",
    type=int,
    default=1000,
)
parser.add_argument(
    "--min_words",
    type=int,
    default=32,
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=40,
)
args = parser.parse_args()

NLP = spacy.load("en_core_web_sm")

EXCLUDED_FAMILIES = {"entity", "avgentity"}
LFTK_FEATURES = [
    elem["key"]
    for elem in lftk.search_features()
    if elem.get("formulation") == "foundation"
    and elem["family"] not in EXCLUDED_FAMILIES
]


def flatten(lst: List):
    """Flatten a list of lists into a single list."""
    return [item for sublist in lst for item in sublist]


def load_input_dataset(
    text_column: str,
    num_samples: int,
    input_path: str = None,
    hf_dataset: str = None,
    hf_split: str = "train",
) -> List[str]:
    """Load text from a local JSONL file or a HuggingFace dataset and sample documents.

    Args:
        text_column: Name of the column containing text.
        num_samples: Target number of documents to keep.
        input_path: Path to the input JSONL file. Mutually exclusive with hf_dataset.
        hf_dataset: HuggingFace dataset ID. Mutually exclusive with input_path.
        hf_split: HuggingFace dataset split to load.

    Returns:
        A list of sampled text strings.
    """
    if input_path is not None:
        df = pd.read_json(input_path, lines=True)
        dataset = df[text_column].explode().tolist()
    else:
        from datasets import load_dataset

        ds = load_dataset(hf_dataset, split=hf_split)
        dataset = ds[text_column]

    dataset = [s.strip() for s in dataset if isinstance(s, str)]
    # Oversample a bit to ensure we have enough data after filtering short docs
    sample_size = min(len(dataset), num_samples + int(num_samples * 0.20))
    dataset = random.sample(dataset, sample_size)
    return dataset

def extract_features(documents: List[str]):
    """Extracts foundation probing features from a list of documents."""
    docs = [NLP(text) for text in documents]
    docs = list(
        compress(docs, [len(doc) >= args.min_words for doc in docs])
    )

    LFTK = lftk.Extractor(docs=docs)
    LFTK.customize(stop_words=True, punctuations=True)
    extracted_features = LFTK.extract(features=LFTK_FEATURES)

    data = []
    for i, features in enumerate(extracted_features):
        features["text"] = docs[i].text
        data.append(features)

    return data

def main():
    """Load input dataset, extract foundation features in parallel, and save results."""
    dataset = load_input_dataset(
        text_column=args.text_column,
        num_samples=args.num_samples,
        input_path=args.input_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
    )
    batches = [
        dataset[i:i+args.num_docs_per_worker]
        for i in range(0, len(dataset), args.num_docs_per_worker)
    ]
    print(colored(f"len(dataset)={len(dataset)}; len(batches)={len(batches)}", "green"))
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(extract_features, batches), total=len(batches)))

    results = flatten(results)
    results = results[:args.num_samples]
    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_json(args.output_path, lines=True, orient="records")
    return 0

if __name__ == "__main__":
    sys.exit(main())
