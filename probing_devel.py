

import json
import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
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
TRAIN_BATCH_SIZE = 512

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
        
def run_logistic_regression(
    input_dim,
    num_classes,
    train_dataloader, 
    val_dataloader,
    test_dataloader,
    max_epoch=200,
    tenacity=5,
    L2reg=[10**t for t in range(-5, -1)],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_models = [None for _ in range(len(L2reg))]
    best_scores = [float("-inf") for _ in range(len(L2reg))]
    for i, reg in enumerate(L2reg):
        model = LogisticRegression(input_dim, num_classes)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=reg)
        early_stop_count = 0
        for _ in tqdm(range(max_epoch)):
            
            model.train()
            for X, y in train_dataloader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                loss = torch.nn.functional.cross_entropy(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                val_acc = 0.
                num_samples = 0
                for X, y in val_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y_pred = model(X)
                    val_acc += torch.eq(y_pred.argmax(dim=1), y).float().sum()
                    num_samples += X.size(0)
                val_acc /= num_samples

                if val_acc > best_scores[i]:
                    best_scores[i] = val_acc.item()
                    best_models[i] = {k:v.to("cpu") for k, v in deepcopy(model.state_dict()).items()}
                else:
                    early_stop_count += 1
                    if early_stop_count > tenacity:
                        break
            
    best_idx = np.argmax(best_scores)
    model.load_state_dict(best_models[best_idx], strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_acc = 0.
        num_samples = 0
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            test_acc += torch.eq(y_pred.argmax(dim=1), y).float().sum()
            num_samples += X.size(0)
        test_acc /= num_samples

    return test_acc.item()

def get_scores_savepath():
    if not IS_LUAR:
        model_str = "sbert"
    else:
        model_str = os.path.basename(HF_OR_MODEL_PATH)
        if model_str == "":
            model_str = os.path.basename(os.path.dirname(HF_OR_MODEL_PATH))
    
    scores_savepath = f"./outputs_probing_devel/{model_str}_results.json"
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
    return torch.FloatTensor(embeddings)

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
        train_y = torch.LongTensor(train["label"].tolist())
        val_text = val["text"].tolist()
        val_y = torch.LongTensor(val["label"].tolist())
        test_text = test["text"].tolist()
        test_y = torch.LongTensor(test["label"].tolist())

        train_X = extract_embeddings(model, tokenizer, train_text, batch_size=BATCH_SIZE)
        val_X = extract_embeddings(model, tokenizer, val_text, batch_size=BATCH_SIZE)
        test_X = extract_embeddings(model, tokenizer, test_text, batch_size=BATCH_SIZE)

        train_dataloader = DataLoader(
            TensorDataset(train_X, train_y),
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=4
        )
        val_dataloader = DataLoader(
            TensorDataset(val_X, val_y),
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=4
        )
        test_dataloader = DataLoader(
            TensorDataset(test_X, test_y),
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=4
        )

        test_acc = run_logistic_regression(
            input_dim=train_X.shape[1], 
            num_classes=len(np.unique(train_y)),
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            max_epoch=200,
            tenacity=5,
            L2reg=[10**t for t in range(-5, -1)],
        )
        scores[os.path.basename(dataset).split(".")[0]] = test_acc
        print(scores)
        with open(scores_savepath, "w") as f:
            json.dump(scores, f, indent=4)

        if DEBUG and i > 1:
            break
        
    return 0

if __name__ == "__main__":
    sys.exit(main())