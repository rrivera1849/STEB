import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from models.base import STEBModel
from typing import List

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class HFModel(STEBModel):
    def __init__(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_single(self, texts: List[str], batch_size: int) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            max_length = 512
            tokenized_batch = self.tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized_batch = {
                k: v.to(self.device)
                for k, v in tokenized_batch.items()
            }
            with torch.no_grad():
                features = self.model(**tokenized_batch)
                features = mean_pooling(features, tokenized_batch["attention_mask"])
                features = features.detach().cpu().numpy()
            all_embeddings.append(features)
        return np.concatenate(all_embeddings)

    def embed_multiple(self, episodes: List[List[str]], batch_size: int) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]

            # Flatten the batch of episodes into a single list of texts
            texts = [text for episode in batch for text in episode]

            max_length = 512
            tokenized_batch = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized_batch = {
                k: v.to(self.device)
                for k, v in tokenized_batch.items()
            }
            with torch.no_grad():
                features = self.model(**tokenized_batch)
                features = mean_pooling(features, tokenized_batch["attention_mask"])

                # Reshape the features back to the episode structure and average
                episode_size = len(batch[0])
                features = features.reshape(len(batch), episode_size, -1)
                features = features.mean(dim=1)

                features = features.detach().cpu().numpy()
            all_embeddings.append(features)
        return np.concatenate(all_embeddings)
