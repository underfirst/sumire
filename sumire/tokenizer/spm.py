from os import environ, makedirs, remove
from pathlib import Path
from shutil import copy
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional, Union
from uuid import uuid4

import sentencepiece
from loguru import logger

from sumire.tokenizer.base import BaseTokenizer, TokenizerInputs, TokenizerOutputs

ModelType = Literal["unigram", "bpe", "char", "word"]


class SentencePieceTokenizer(BaseTokenizer):
    """
    Tokenizer class using SentencePiece for text tokenization.

    Args:
        vocab_size (int, optional): Vocabulary size. Default is 32000.
        model_type (ModelType, optional): Type of SentencePiece model. Default is "unigram".
        character_coverage (float, optional): Character coverage for SentencePiece model. Default is 0.995.
        spm_model (sentencepiece.SentencePieceProcessor, optional): Pre-trained SentencePiece model. Default is None.
    """
    tokenizer_file_prefix = "sentencepiece"
    model_cache_dir = Path(environ["HOME"]) / f".cache/sumire/sentencepiece/{uuid4()}"

    def __init__(
        self,
        vocab_size: int = 32000,
        model_type: ModelType = "unigram",
        character_coverage: float = 0.995,
        spm_model: Optional[sentencepiece.SentencePieceProcessor] = None,
            *args, **kwargs
    ):
        super().__init__()
        self.init_args["vocab_size"] = vocab_size
        self.init_args["model_type"] = model_type
        self.init_args["character_coverage"] = character_coverage

        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage

        if model_type == "word":
            logger.warning("model_type=word need pre-tokenization.")

        self.tokenizer = spm_model

    def fit(self, inputs: List[str]) -> None:
        """
        Fits the tokenizer on a list of texts.

        Args:
            inputs (List[str]): List of texts to fit the tokenizer on.
        """
        with TemporaryDirectory() as dir_path:
            corpus_path = Path(dir_path) / "corpus.txt"
            with open(corpus_path, "w") as f:
                for line in inputs:
                    f.write(f"{line}\n")

            sentencepiece.SentencePieceTrainer.train(
                input=corpus_path,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                model_prefix=self.tokenizer_file_prefix,
            )
            makedirs(self.model_cache_dir, exist_ok=True)
            for ext in [".model", ".vocab"]:
                copy(self.tokenizer_file_prefix + ext, self.model_cache_dir / (self.tokenizer_file_prefix + ext))
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(self.model_cache_dir / (self.tokenizer_file_prefix + ".model")))

    def tokenize(self, inputs: TokenizerInputs) -> TokenizerOutputs:
        """
        Tokenizes input text using SentencePiece.

        Args:
            inputs (str or List[str]): Text or list of texts to tokenize.

        Returns:
            TokenizerOutputs: Tokenized texts as strings.
        """
        if self.tokenizer is None:
            raise AssertionError("SentencePieceTokenizer need fit(List[str]) before using.")
        if isinstance(inputs, str):
            inputs = [inputs]
        return self.tokenizer.encode(inputs, out_type=str)

    def encode(self, inputs: TokenizerInputs) -> List[List[int]]:
        """
        Encodes input text using SentencePiece.

        Args:
            inputs (str or List[str]): Text or list of texts to encode.

        Returns:
            List[List[int]]: Encoded tokens as lists of integers.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        return self.tokenizer.encode(inputs)

    def decode(self, inputs: Union[List[int], List[List[int]]]) -> List[str]:
        """
        Decodes input tokens using SentencePiece.

        Args:
            inputs (List[int] or List[List[int]]): Encoded tokens to decode.

        Returns:
            List[str]: Decoded texts as strings.
        """
        if isinstance(inputs[0], int):
            inputs = [inputs]
        return self.tokenizer.decode(inputs)

    def save_pretrained(self, path: Union[str, Path]):
        """
        Saves tokenizer configuration to the specified path.

        Args:
            path (str or Path): Directory path to save the configuration.
        """
        self._save_pretrained(path, self.__class__.__name__)
        path = Path(path)
        for ext in [".model", ".vocab"]:
            model_file_name = self.tokenizer_file_prefix + ext
            copy(self.model_cache_dir / model_file_name, path / model_file_name)
            remove(self.model_cache_dir / model_file_name)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Loads tokenizer from a saved configuration.

        Args:
            path (str or Path): Directory path to the saved configuration.

        Returns:
            SentencePieceTokenizer: Tokenizer instance initialized with the saved configuration.
        """
        configs = cls._load_config(path)
        path = Path(path)
        if (path / f"{cls.tokenizer_file_prefix}.model").exists() and (
            path / f"{cls.tokenizer_file_prefix}.vocab"
        ).exists():
            spm_model = sentencepiece.SentencePieceProcessor(model_file=str(path / f"{cls.tokenizer_file_prefix}.model"))
            init_args = configs["init_args"]
            init_args["spm_model"] = spm_model
            return cls(**init_args)
