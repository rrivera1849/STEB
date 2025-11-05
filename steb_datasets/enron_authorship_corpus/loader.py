
import os
import json
from collections import defaultdict
from typing import Dict, List
import random
import spacy
from termcolor import colored
from tqdm import tqdm

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
