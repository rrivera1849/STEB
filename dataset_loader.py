
import json
import os
import configparser
from collections import defaultdict
from functools import partial
from typing import Dict, List

import pandas as pd
import spacy
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm

from utils import *

config = configparser.ConfigParser()
config.read('config.ini')
PROCESSED_DATA_DIR = config['Application_Paths']['processed_dataset_dir']
CACHE_DIR = config['Application_Paths']['cache_dir']

def blog_authorship_labelgetter(age):
    if age < 18:
        return 0
    elif age < 28:
        return 1
    else:
        return 2

CUSTOM_TRANSFORMS = {
    "blog_authorship_corpus": blog_authorship_labelgetter
}

def record_handler(example, text_getter, label_getter, label_transform=None):
    text = example[text_getter]
    label = example[label_getter]
    if isinstance(label, list):
        label = label[0] if label else None
    
    if text is None or label is None:
        return None

    if label_transform:
        label = label_transform(label)
    
    return {"text": text, "label": label}

def load_hate_speech_dataset(path: str):
    """
        GitHub: https://github.com/Vicomtech/hate-speech-dataset
        Paper: https://aclanthology.org/W18-51.pdf
    """
    records = []
    
    annotations = pd.read_csv(os.path.join(path, "annotations_metadata.csv"))
    test_dir = os.path.join(path, "sampled_test")
    for fname in os.listdir(test_dir):
        text = open(os.path.join(test_dir, fname), "r").read()
        file_id = os.path.splitext(fname)[0]
        label = annotations[annotations["file_id"] == file_id]["label"].iloc[0]
        
        records.append({
            "text": text,
            "label": label,
        })
        
    return records

def load_hate_speech_and_offensive_language(path: str):
    """
        GitHub: https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master
        Paper: https://ojs.aaai.org/index.php/ICWSM/article/view/14955
    """
    records = []
    df = pd.read_csv(os.path.join(path, "labeled_data.csv"))
    for _, row in df.iterrows():
        records.append({
            "text": row["tweet"],
            "label": row["class"],
        })
    return records

def load_enron_authorship_corpus(path: str):
    """
        URL: https://data.mendeley.com/datasets/n77w7mygwg/1
    """
    dirnames = os.listdir(path)
    
    def _to_records(author_to_sentences: Dict[str, List[str]]):
        records = []
        for author, emails in author_to_sentences.items():
            for email in emails:
                records.append({
                    "text": email,
                    "label": author,
                })
        return records

    if os.path.exists(os.path.join(path, "enron_authorship_corpus.json")):
        print(colored("Loading enron_authorship_corpus", "green"))
        author_to_sentences = json.loads(open(os.path.join(path, "enron_authorship_corpus.json"), "r").read())
        return _to_records(author_to_sentences)

    print(colored("Creating enron_authorship_corpus", "yellow"))
    author_to_sentences: Dict[str, List[str]] = defaultdict(list)
    nlp = spacy.load("en_core_web_sm")
    
    for dirname in tqdm(dirnames):
        if not os.path.isdir(os.path.join(path, dirname)): continue
        
        all_sentences = []
        emails = os.listdir(os.path.join(path, dirname))
        for email in emails:
            text = open(os.path.join(path, dirname, email), "r").read()
            sentences = [sent.text for sent in nlp(text).sents]
            all_sentences.extend(sentences)

        random.shuffle(all_sentences)
        author_to_sentences[dirname[:-3]] = all_sentences

    with open(os.path.join(path, "enron_authorship_corpus.json"), "w") as f:
        f.write(json.dumps(author_to_sentences))
        
    return _to_records(author_to_sentences)

CUSTOM_LOADERS = {
    "hate_speech": load_hate_speech_dataset,
    "hate_speech_and_offensive_language": load_hate_speech_and_offensive_language,
    "enron_authorship_corpus": load_enron_authorship_corpus,
}

class DatasetLoader(object):
    """This class is responsible for loading the dataset and creating the episodes.
    """
    def __init__(
        self,
        dataset_name: str,
        episode_size: int = 5,
        n_episodes_per_class: int = 50,
        force_reload: bool = False,
    ):
        self.dataset_name = dataset_name
        self.episode_size = episode_size
        self.n_episodes_per_class = n_episodes_per_class
        self.force_reload = force_reload
        self.config = self._load_config()

    def _load_config(self):
        config_path = os.path.join("datasets", self.dataset_name, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found for dataset: {self.dataset_name}")
        with open(config_path, "r") as f:
            return json.load(f)

    def load(self):
        dataset_path = self._get_dataset_path()
        
        if os.path.exists(dataset_path) and not self.force_reload:
            print(colored(f"Loading dataset from {dataset_path}", "green"))
            return json.loads(open(dataset_path, "r").read())

        if self.config["type"] == "huggingface":
            loader_kwargs = self.config["loader_kwargs"]
            loader_kwargs["cache_dir"] = CACHE_DIR
            dataset_iter = load_dataset(**loader_kwargs)
        elif self.config["type"] == "custom":
            loader_fn = CUSTOM_LOADERS[self.dataset_name]
            dataset_iter = loader_fn(self.config["data_dir"])
        else:
            raise ValueError(f"Unknown dataset type: {self.config['type']}")

        text_getter = self.config["record_handler"]["text_getter"]
        label_getter = self.config["record_handler"]["label_getter"]
        label_transform = CUSTOM_TRANSFORMS.get(self.dataset_name)

        handler = partial(record_handler, text_getter=text_getter, label_getter=label_getter, label_transform=label_transform)
        
        N = self.episode_size * self.n_episodes_per_class
        dataset: Dict[str, List[str]] = defaultdict(list)
        valid_labels = self.get_valid_labels(dataset_iter, handler)

        for example in dataset_iter:
            record = handler(example)
            if record is None or record["label"] not in valid_labels:
                continue
            elif len(dataset[record["label"]]) >= N:
                continue
            dataset[record["label"]].append(record["text"])

        dataset = {k: v for k, v in dataset.items() if len(v) == N}
        with open(dataset_path, "w") as f:
            print(f"Saving dataset to {dataset_path}")
            f.write(json.dumps(dataset))
        return dataset
    
    def _get_dataset_path(self):
        base_str = f"{os.path.basename(self.dataset_name)}_{self.n_episodes_per_class}_{self.episode_size}"
        return os.path.join(PROCESSED_DATA_DIR, base_str + ".json")
    
    def get_valid_labels(self, dataset_iter, handler):
        """Gets all the labels for classes that surpass:
            - self.episode_size * self.n_episodes_per_class
        """
        N = self.episode_size * self.n_episodes_per_class

        label_to_count: Dict[str, int] = defaultdict(int)
        for example in dataset_iter:
            record = handler(example)
            if record is None:
                continue
            label_to_count[record["label"]] += 1

        label_to_count = {k: v for k, v in label_to_count.items() if v >= N}
        valid_labels = list(label_to_count.keys())

        return valid_labels
