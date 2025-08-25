
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from joblib import Memory
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored
from transformers import AutoModel, AutoTokenizer, set_seed

from dataset_loader import DatasetLoader

parser = ArgumentParser()
parser.add_argument("--model_name_or_path",
                    default="rrivera1849/LUAR-CRUD",
                    help="Model name (HF ID), or local path to the model.")
parser.add_argument("--dataset",
                    default="billray110/corpus-of-diverse-styles",
                    help="Dataset to evaluate on.")
parser.add_argument("--cache_dir",
                    default="/tmp/riverasoto1",
                    help="Location to a temporary cache directory.")
parser.add_argument("-e", "--episode_sizes", type=int, action="append", default=None, required=True,
                    help="Number of atomic units to form writing sample.")
parser.add_argument("--n_episodes_per_class", type=int, default=50,
                    help="Number of examples per class.")
parser.add_argument("--force_reload", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=43)
FLAGS = parser.parse_args()
set_seed(FLAGS.seed)

def calculate_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_scores_path(episode_size):
    model_str = os.path.basename(FLAGS.model_name_or_path)
    if model_str == "":
        model_str = os.path.basename(os.path.dirname(FLAGS.model_name_or_path))
    
    dset_str = os.path.basename(FLAGS.dataset)
    scores_path = f"./outputs/{dset_str}/{model_str}/{episode_size}_{FLAGS.n_episodes_per_class}"
    return scores_path

def is_luar():
    return "LUAR" in FLAGS.model_name_or_path

def main():
    if torch.cuda.is_available():
        print(colored("GPU information", "green"))
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print(colored("WARNING: No GPU detected", "red"))
        device = torch.device("cpu")

    memory = Memory(FLAGS.cache_dir, verbose=1)
    memory.clear()

    print(FLAGS.model_name_or_path)
    model = AutoModel.from_pretrained(
        FLAGS.model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        FLAGS.model_name_or_path, 
        trust_remote_code=True
    )
    model.eval()
    model.to(device)

    @memory.cache
    def extract_features(dataset, episode_size, n_episodes_per_class):
        X, y = [], []
        
        for label, episodes in dataset.items():
            dataset[label] = [episodes[i:i+episode_size] for i in range(0, len(episodes), episode_size)]
            assert len(dataset[label]) == n_episodes_per_class
            assert all([len(episode) == episode_size for episode in dataset[label]])
        
        for label, episodes in dataset.items():
            for episode in episodes:
                # default used in the TACL experiments, should be using model's maximum capacity
                max_length = 512
                tokenized_episode = tokenizer(
                    episode,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokenized_episode = {
                    k: v.to(device)
                    for k, v in tokenized_episode.items()
                }

                if not is_luar():
                    with torch.no_grad():
                        features = model(**tokenized_episode)
                        features = mean_pooling(features, tokenized_episode["attention_mask"])
                        features = features.mean(dim=0, keepdim=True).detach().cpu().numpy()
                else:
                    tokenized_episode["input_ids"] = \
                        tokenized_episode["input_ids"].reshape(1, episode_size, max_length)
                    tokenized_episode["attention_mask"] = \
                        tokenized_episode["attention_mask"].reshape(1, episode_size, max_length)
                    with torch.no_grad():
                        features = model(
                            input_ids=tokenized_episode["input_ids"],
                            attention_mask=tokenized_episode["attention_mask"]
                        ).detach().cpu().numpy()
                        
                X.append(features)
                y.append(label)
        return X, y

    for episode_size in FLAGS.episode_sizes:
    
        dataset = DatasetLoader(
            dataset_name=FLAGS.dataset,
            episode_size=episode_size,
            n_episodes_per_class=FLAGS.n_episodes_per_class,
            force_reload=FLAGS.force_reload,
        ).load()
        
        if len(dataset) <= 0:
            continue

        X, y = extract_features(dataset, episode_size, FLAGS.n_episodes_per_class)
        scores = cosine_similarity(
            np.array(X).reshape(-1, X[0].shape[-1]), 
            np.array(X).reshape(-1, X[0].shape[-1])
        ) # pairwise cosine similarities
        labels = np.array(y).reshape(-1, 1) == np.array(y).reshape(1, -1)
        scores = scores[np.triu_indices(scores.shape[0], k=1)]
        labels = labels[np.triu_indices(labels.shape[0], k=1)]

        eer = calculate_eer(labels, scores)
        auc = roc_auc_score(labels, scores)
        auc_threshold = roc_auc_score(labels, scores, max_fpr=0.01)
        
        scores_path = get_scores_path(episode_size)
        os.makedirs(scores_path, exist_ok=True)
        with open(os.path.join(scores_path, "scores.npy"), "wb") as ouf:
            np.savez(ouf, scores=np.array(scores), labels=np.array(labels))
        with open(os.path.join(scores_path, "metrics.json"), "w+") as ouf:
            ouf.write(
                json.dumps(
                    {
                        "eer": eer,
                        "auc": auc,
                        "auc_threshold": auc_threshold
                    }
                )
            )
        print(f"{FLAGS.dataset}; episode size: {episode_size}; N: {FLAGS.n_episodes_per_class}; EER: {eer}; AUC: {auc}, AUC (FPR <= 0.01): {auc_threshold}")

    return 0

if __name__ == "__main__":
    sys.exit(main())