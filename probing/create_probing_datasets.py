
import os
import sys
from argparse import ArgumentParser
from itertools import compress
from multiprocessing import Pool
from typing import List

import lftk
import pandas as pd
import spacy
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm

from utils import PROCESSED_DATA_DIR

parser = ArgumentParser()
parser.add_argument("--num_docs_per_worker", type=int, default=1000)
parser.add_argument("--min_words", type=int, default=15)
parser.add_argument("--max_words", type=int, default=28)
parser.add_argument("--num_workers", type=int, default=40)
args = parser.parse_args()

NLP = spacy.load("en_core_web_sm")
LFTK_SYNTAX_FEATURES = lftk.search_features(domain="syntax", family="partofspeech")
LFTK_SYNTAX_FEATURES = [elem["key"] for elem in LFTK_SYNTAX_FEATURES]

def flatten(list: List):
    return [item for sublist in list for item in sublist]

def load_dataset():
    dataset = pd.read_json(
        "/data1/yubnub/data/iur_dataset/train.jsonl", 
        lines=True,
    )
    dataset = dataset[["author_id", "syms"]].explode("syms")["syms"].tolist()
    return dataset

def extract_features(documents: List[str]):
    """Extracts syntax features from a list of documents.
    """
    docs = [NLP(text) for text in documents]
    docs = list(
        compress(docs, [len(doc) >= args.min_words and len(doc) <= args.max_words for doc in docs])
    )

    LFTK = lftk.Extractor(docs=docs)
    LFTK.customize(stop_words=True, punctuations=True)
    extracted_features = LFTK.extract(features=LFTK_SYNTAX_FEATURES)

    data = []
    for i, features in enumerate(extracted_features):
        features["text"] = docs[i].text
        data.append(features)

    return data

def main():
    dataset = load_dataset()
    batches = [
        dataset[i:i+args.num_docs_per_worker] 
        for i in range(0, len(dataset), args.num_docs_per_worker)
    ]
    print(colored(f"len(dataset)={len(dataset)}; len(batches)={len(batches)}", "green"))
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(extract_features, batches), total=len(batches)))
    
    df = pd.DataFrame(flatten(results))
    df.to_json(
        os.path.join(PROCESSED_DATA_DIR, "probing/reddit.jsonl"),
        lines=True, orient="records"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())