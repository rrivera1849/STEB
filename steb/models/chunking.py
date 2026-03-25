import spacy
from typing import List

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def sentencize(text: str) -> List[str]:
    """
    Splits text into sentences using spacy.

    Args:
        text: The input text to split.

    Returns:
        A list of sentence strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return [s for s in sentences if s]


def chunk_text(text: str, tokenizer, max_length: int) -> List[str]:
    """
    Splits a text into chunks that fit within the model's max token length.

    If the text already fits within max_length tokens, returns it as-is.
    Otherwise, splits into sentences and greedily groups them into chunks.

    Args:
        text: The input text to chunk.
        tokenizer: A HuggingFace tokenizer for token counting.
        max_length: The maximum number of tokens per chunk (including special tokens).

    Returns:
        A list of text chunks, each fitting within max_length tokens.
    """
    if not text or not text.strip():
        return [text]

    # Early exit: if the full text fits, no chunking needed
    if len(tokenizer.encode(text, add_special_tokens=True)) <= max_length:
        return [text]

    sentences = sentencize(text)
    if not sentences:
        return [text]

    # Compute token budget excluding special tokens
    overhead = len(tokenizer.encode("", add_special_tokens=True))
    budget = max_length - overhead

    # Count tokens per sentence (without special tokens)
    sentence_token_counts = [
        len(tokenizer.encode(sent, add_special_tokens=False))
        for sent in sentences
    ]

    chunks = []
    current_sentences = []
    current_token_count = 0

    for sent, token_count in zip(sentences, sentence_token_counts):
        # If adding this sentence would exceed budget, finalize current chunk
        if current_sentences and current_token_count + token_count > budget:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_token_count = 0

        current_sentences.append(sent)
        current_token_count += token_count

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
