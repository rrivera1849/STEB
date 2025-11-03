import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from models.base import STEBModel
from typing import List

class LUARModel(STEBModel):
    def __init__(self, model_name_or_path: str):
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_single(self, texts: List[str], batch_size: int) -> np.ndarray:
        # Treat single texts as episodes of size 1
        episodes = [[text] for text in texts]
        return self.embed_multiple(episodes, batch_size)

    def embed_multiple(self, episodes: List[List[str]], batch_size: int) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]

            # Flatten the batch of episodes into a single list of texts
            texts = [text for episode in batch for text in episode]

            max_length = 512
            episode_size = len(batch[0])
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

            tokenized_batch["input_ids"] = \
                tokenized_batch["input_ids"].reshape(len(batch), episode_size, max_length)
            tokenized_batch["attention_mask"] = \
                tokenized_batch["attention_mask"].reshape(len(batch), episode_size, max_length)

            with torch.no_grad():
                features = self.model(
                    input_ids=tokenized_batch["input_ids"],
                    attention_mask=tokenized_batch["attention_mask"]
                ).detach().cpu().numpy()
            all_embeddings.append(features)
        return np.concatenate(all_embeddings)
