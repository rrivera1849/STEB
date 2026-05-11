"""EncT5 tokenizer for LISA models.

Source: https://ajayp.app/posts/2023/11/learning-interpretable-embeddings-via-llms/
"""
from typing import Any, Dict, List, Optional

from transformers import T5Tokenizer


class EncT5Tokenizer(T5Tokenizer):
    """T5 tokenizer variant that adds BOS/EOS special tokens."""

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the EncT5Tokenizer.

        Args:
            vocab_file: Path to the SentencePiece vocabulary file.
            bos_token: Beginning of sequence token.
            eos_token: End of sequence token.
            unk_token: Unknown token.
            pad_token: Padding token.
            extra_ids: Number of extra IDs for sentinel tokens.
            additional_special_tokens: Additional special tokens.
            sp_model_kwargs: SentencePiece model keyword arguments.
            **kwargs: Additional keyword arguments.
        """
        sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=sp_model_kwargs,
            **kwargs,
        )

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Retrieve sequence ids from a token list that has no special tokens added.

        Args:
            token_ids_0: List of IDs.
            token_ids_1: Optional second list of IDs for sequence pairs.
            already_has_special_tokens: Whether the token list already has special tokens.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """Create a mask from two sequences for sequence-pair classification.

        Args:
            token_ids_0: List of IDs.
            token_ids_1: Optional second list of IDs for sequence pairs.

        Returns:
            List of zeros (T5 does not use token type ids).
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """Build model inputs with BOS/EOS special tokens.

        Args:
            token_ids_0: List of IDs to which the special tokens will be added.
            token_ids_1: Optional second list of IDs for sequence pairs.

        Returns:
            List of input IDs with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
