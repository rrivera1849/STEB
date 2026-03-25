import pytest
from transformers import AutoTokenizer

from steb.models.chunking import sentencize, chunk_text


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


class TestSentencize:
    def test_basic_sentences(self):
        text = "Hello world. How are you? Fine thanks."
        sentences = sentencize(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "Fine thanks."

    def test_single_sentence(self):
        text = "Just one sentence here."
        sentences = sentencize(text)
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence here."

    def test_empty_string(self):
        sentences = sentencize("")
        assert sentences == []


class TestChunkText:
    def test_short_text_no_chunking(self, tokenizer):
        text = "This is a short sentence."
        chunks = chunk_text(text, tokenizer, max_length=512)
        assert chunks == [text]

    def test_empty_text(self, tokenizer):
        chunks = chunk_text("", tokenizer, max_length=512)
        assert chunks == [""]

    def test_multi_sentence_chunking(self, tokenizer):
        # Create a text with many sentences that exceeds a small max_length
        sentences = [f"This is sentence number {i} and it contains some words." for i in range(20)]
        text = " ".join(sentences)
        max_length = 40

        chunks = chunk_text(text, tokenizer, max_length=max_length)

        assert len(chunks) > 1
        # Verify each chunk fits within the token limit
        for chunk in chunks:
            token_count = len(tokenizer.encode(chunk, add_special_tokens=True))
            assert token_count <= max_length, (
                f"Chunk has {token_count} tokens, exceeding max_length={max_length}"
            )
        # Verify no text is lost: all original sentences appear in some chunk
        joined = " ".join(chunks)
        for sent in sentences:
            assert sent in joined

    def test_single_long_sentence(self, tokenizer):
        # A single sentence that exceeds max_length -- returns as single chunk (truncated at tokenization)
        long_sentence = "word " * 200
        chunks = chunk_text(long_sentence.strip(), tokenizer, max_length=50)
        assert len(chunks) == 1

    def test_chunk_token_counts(self, tokenizer):
        text = ". ".join([f"Sentence {i} has several tokens in it" for i in range(30)]) + "."
        max_length = 32

        chunks = chunk_text(text, tokenizer, max_length=max_length)

        for chunk in chunks:
            token_count = len(tokenizer.encode(chunk, add_special_tokens=True))
            # Chunks with a single sentence may exceed if that sentence alone is too long
            # but multi-sentence chunks should respect the limit
            sentences_in_chunk = sentencize(chunk)
            if len(sentences_in_chunk) > 1:
                assert token_count <= max_length

