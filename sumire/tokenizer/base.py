import json
from abc import ABC, abstractmethod
from os import makedirs
from pathlib import Path
from typing import Dict, List, TypeVar, Union

from loguru import logger

TokenizerInputs = Union[str, List[str]]
TokenizerOutputs = List[List[str]]


class BaseTokenizer(ABC):
    tokenizer_config_file = "tokenizer_config.json"

    def __init__(self, *args, **kwargs):
        self.init_args = dict()

    def fit(self, inputs: TokenizerInputs) -> None:
        """
        Training tokenizer if necessary.
        """
        logger.warning("This tokenizer did not implement fit() method.")

    @abstractmethod
    def tokenize(self, inputs: TokenizerInputs) -> TokenizerOutputs:
        """
        Tokenizes the input text or list of texts.

        Args:
            inputs (TokenizerInputs): The input text or a list of texts to tokenize.

        Returns:
            TokenizerOutputs: A list of lists, where each inner list represents tokenized words for a single input text.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def save_pretrained(self, path: Union[str, Path]):
        """
        Saves the pretrained tokenizer to the specified path.

        Args:
            path (str or Path): The directory path where the pretrained tokenizer will be saved.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Loads a pretrained tokenizer from the specified path.

        Args:
            path (str or Path): The directory path where the pretrained tokenizer is saved.

        Returns:
            BaseTokenizer: An instance of the pretrained tokenizer.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError()

    def _save_pretrained(self, path: Union[str, Path], tokenizer_class: str):
        """
        Common save_pretrained operation.
        In the Tokenizer class, basically saving tokenizer initialization arguments to "tokenizer_config.json".

        Args:
            path (str or Path): The directory path where the pretrained tokenizer will be saved.
        """
        path = Path(path)
        if path.exists() and path.is_dir():
            path = path / self.tokenizer_config_file
        elif path.exists() and path.is_file():
            raise ValueError("A file already exists. Pass a directory path.")
        else:
            makedirs(path)
            path = path / self.tokenizer_config_file

        with open(path, "w") as f:
            data = {"tokenizer_class": tokenizer_class, "init_args": self.init_args}
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def _load_config(cls, path: Union[str, Path]) -> Dict:
        """
        Loading tokenizer_config.json to load a tokenizer instance.

        Args:
            path (str or Path): The directory path where the pretrained tokenizer is saved.
        """
        logger.info(f"Load {path}")
        path = Path(path)
        if not (path / cls.tokenizer_config_file).exists():
            raise ValueError("Invalid pre-trained tokenize path.")
        with open(path / cls.tokenizer_config_file) as f:
            configs = json.load(f)
        return configs


TokenizerType = TypeVar("TokenizerType", bound=BaseTokenizer)
