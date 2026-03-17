
from copy import deepcopy
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import Task

class LogisticRegression(nn.Module):
    """Simple Linear -> Sigmoid classifier.
       NOTE: We should NOT make this a stronger model.
    """
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
    """Following SentEval:
    https://github.com/facebookresearch/SentEval/blob/main/senteval/tools/validation.py#L202
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    best_models = [None for _ in range(len(L2reg))]
    best_scores = [float("-inf") for _ in range(len(L2reg))]
    for i, reg in enumerate(L2reg):
        model = LogisticRegression(input_dim, num_classes)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=reg)
        early_stop_count = 0

        for _ in range(max_epoch):

            # Training loop:
            model.train()
            for X, y in train_dataloader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                loss = torch.nn.functional.cross_entropy(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation loop:
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
            
    # Use best model to evaluate on test set:
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
        if num_samples > 0:
            test_acc /= num_samples
        else:
            test_acc = torch.tensor(0.0)

    return test_acc.item()

class ProbingTask(Task):
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: List[Dict[str, Any]]
    ) -> Dict[str, float]:

        # labels is a list of dictionaries with keys: "label", "split"
        # "label" is a list of integers, one for each task
        # "split" is a list of strings, one for each task

        BATCH_SIZE = 512
        num_tasks = len(labels[0]["label"])
        results = {}
        
        for task_idx in range(num_tasks):

            # 1. Collect training, validation, and test embeddings and labels
            train_X, train_y = [], []
            val_X, val_y = [], []
            test_X, test_y = [], []
            
            for emb, meta in zip(embeddings, labels):
                l = meta["label"][task_idx]
                s = meta["split"][task_idx]
                
                if l is None or s is None:
                    continue
                
                if s == "train":
                    train_X.append(emb)
                    train_y.append(l)
                elif s == "val":
                    val_X.append(emb)
                    val_y.append(l)
                elif s == "test":
                    test_X.append(emb)
                    test_y.append(l)
            
            train_X = torch.FloatTensor(np.array(train_X))
            train_y = torch.LongTensor(np.array(train_y))
            val_X = torch.FloatTensor(np.array(val_X))
            val_y = torch.LongTensor(np.array(val_y))
            test_X = torch.FloatTensor(np.array(test_X))
            test_y = torch.LongTensor(np.array(test_y))

            train_dataloader = DataLoader(
                TensorDataset(train_X, train_y), 
                batch_size=BATCH_SIZE, 
                num_workers=0
            )
            val_dataloader = DataLoader(
                TensorDataset(val_X, val_y), 
                batch_size=BATCH_SIZE, 
                num_workers=0
            )
            test_dataloader = DataLoader(
                TensorDataset(test_X, test_y), 
                batch_size=BATCH_SIZE, 
                num_workers=0
            )
            
            num_classes = len(np.unique(train_y))
                
            acc = run_logistic_regression(
                input_dim=train_X.shape[1],
                num_classes=num_classes,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                max_epoch=200,
                tenacity=5,
                L2reg=[10**t for t in range(-5, -1)],
            )
            results[f"task_{task_idx}"] = acc
            
        if results:
            results["average"] = np.mean(list(results.values()))
            
        return results
