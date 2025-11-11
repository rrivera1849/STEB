

import json
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from termcolor import colored
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from utils import PROCESSED_DATA_DIR

HF_OR_MODEL_PATH = sys.argv[1]
DATASET_PATH = os.path.join(
    PROCESSED_DATA_DIR, "probing", "reddit"
)
SEED = 43
IS_LUAR = "LUAR" in HF_OR_MODEL_PATH or "disentangle" in HF_OR_MODEL_PATH
DATASET_DIR = os.path.join(PROCESSED_DATA_DIR, "probing", "reddit")
DEBUG = False
BATCH_SIZE = 2048

def get_scores_savepath():
    if not IS_LUAR:
        model_str = "sbert"
    else:
        model_str = os.path.basename(HF_OR_MODEL_PATH)
        if model_str == "":
            model_str = os.path.basename(os.path.dirname(HF_OR_MODEL_PATH))
    
    scores_savepath = f"./outputs_probing/{model_str}_results.json"
    return scores_savepath

# TODO: remove code replication across the code-base
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_embeddings(
    model, tokenizer,
    text: List[str], 
    batch_size: int = 128, 
    max_length: int = 512,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = []
    batches = [text[i:i+batch_size] for i in range(0, len(text), batch_size)]

    for batch in tqdm(batches):
        tokenized_batch = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
        
        if not IS_LUAR:
            with torch.no_grad():
                features = model(**tokenized_batch)
                features = mean_pooling(features, tokenized_batch["attention_mask"]).detach().cpu().numpy()
        else:
            tokenized_batch = {k: v.unsqueeze(1) for k, v in tokenized_batch.items()}

            with torch.no_grad():
                features = model(**tokenized_batch).detach().cpu().numpy()
            
        embeddings.append(features)
    
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

# TODO: nice typing information & comments
def run_logistic_regression(train_X, val_X, test_X, train_y, val_y, test_y, max_iter=2500):
    # mimics SentEval: https://github.com/facebookresearch/SentEval/blob/main/senteval/tools/validation.py#L202
    regs = [2**t for t in range(-2, 4, 1)]
    scores = []
    for reg in regs:
        clf = LogisticRegression(C=reg, max_iter=max_iter, random_state=SEED)
        clf.fit(train_X, train_y)
        scores.append(round(100 * clf.score(val_X, val_y)))
    
    best_reg = np.argmax(scores)

    clf = LogisticRegression(C=regs[best_reg], max_iter=max_iter, random_state=SEED)
    clf.fit(train_X, train_y)
    test_accuracy = round(100 * clf.score(test_X, test_y))
    return test_accuracy

def main():
    datasets = [os.path.join(DATASET_DIR, fname) for fname in os.listdir(DATASET_DIR) if fname.endswith(".jsonl")]
    scores = {}
    scores_savepath = get_scores_savepath()
    print(colored(f"Saves scores to {scores_savepath}", "green"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(HF_OR_MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(HF_OR_MODEL_PATH, trust_remote_code=True)
    model.half()
    model.eval()
    model.to(device)

    for i, dataset in tqdm(enumerate(datasets)):
        df = pd.read_json(dataset, lines=True)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

        # shouldn't be storing these datasets
        num_classes = len(df["label"].unique())
        if num_classes < 2:
            continue
        
        if DEBUG:
            train = df[df["split"] == "train"].iloc[:1000]
            val = df[df["split"] == "val"].iloc[:1000]
            test = df[df["split"] == "test"].iloc[:1000]
        else:
            train = df[df["split"] == "train"]
            val = df[df["split"] == "val"]
            test = df[df["split"] == "test"]

        train_text = train["text"].tolist()
        train_y = train["label"].tolist()
        val_text = val["text"].tolist()
        val_y = val["label"].tolist()
        test_text = test["text"].tolist()
        test_y = test["label"].tolist()
        
        train_X = extract_embeddings(model, tokenizer, train_text, batch_size=BATCH_SIZE)
        val_X = extract_embeddings(model, tokenizer, val_text, batch_size=BATCH_SIZE)
        test_X = extract_embeddings(model, tokenizer, test_text, batch_size=BATCH_SIZE)
        
        acc = run_logistic_regression(train_X, val_X, test_X, train_y, val_y, test_y)

        scores[os.path.basename(dataset).split(".")[0]] = acc

        if DEBUG and i > 1:
            break
        
    print(scores)
    with open(scores_savepath, "w") as f:
        json.dump(scores, f, indent=4)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())