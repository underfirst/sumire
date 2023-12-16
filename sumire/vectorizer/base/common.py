import json
from abc import ABC, abstractmethod
from os import makedirs
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from sumire.tokenizer import TokenizerInputs, TokenizerOutputs, TokenizerType

EncodeTokensOutputs = List[List[Tuple[str, np.array]]]


class BaseVectorizer(ABC):
    vectorizer_config_file = "vectorizer_config.json"

    def __init__(self):
        self.tokenizer = None
        self.init_args = {}
        self.tokenizer: TokenizerType

    @abstractmethod
    def fit(self, texts: Union[str, List[str]], *args, **kwargs) -> None:
        """
        Fits the vectorizer to the input texts.

        Args:
            texts (Union[str, List[str]]): The input texts or a list of texts to fit the vectorizer.

        Returns:
            None
        """
        logger.warning("This vectorizer did not implement fit() method.")

    @abstractmethod
    def transform(self, texts: Union[str, List[str]], *args, **kwargs) -> np.array:
        """
        Transforms the input texts into numerical vectors.

        Args:
            texts (Union[str, List[str]]): The input texts or a list of texts to transform.

        Returns:
            np.array: An array of numerical vectors representing the input texts.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Saves the pretrained vectorizer to the specified directory.

        Args:
            path (str or Path): The directory path where the pretrained vectorizer will be saved.

        Returns:
            None

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Loads a pretrained vectorizer from the specified directory.

        Args:
            path (str or Path): The directory path where the pretrained vectorizer is saved.

        Returns:
            BaseVectorizer: An instance of the pretrained vectorizer.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError()

    def _save_vectorizer_config(self, path: Union[str, Path], vectorizer_class: str, params: Optional[Dict] = None):
        """
        Utility method to save vectorizer config file.
        In the Vectorizer class, saving class name and instance initialization arguments to vectorizer_config.json
        In the Tokenizer class, basically saving tokenizer initialization arguments to "tokenizer_config.json".

        Args:
            path (str or Path): The directory path where the pretrained tokenizer will be saved.
            vectorizer_class (str): pass __class__.__name__.
            params (dict): additional serializable value to save in vectorizer_config.json.
        """
        path = Path(path)
        if path.exists() and path.is_dir():
            path = path / self.vectorizer_config_file
        elif path.exists() and path.is_file():
            raise ValueError("A file already exists. Pass a directory path.")
        else:
            makedirs(path)
            path = path / self.vectorizer_config_file

        with open(path, "w") as f:
            if params is None:
                params = {}
            params["vectorizer_class"] = vectorizer_class
            params["init_args"] = self.init_args
            json.dump(params, f, ensure_ascii=False, indent=2)

    @classmethod
    def _load_vectorizer_config(cls, path: str):
        """
        Loading vectorizer_config.json to load a vectorizer instance.

        Args:
            path (str or Path): The directory path where the pretrained vectorizer is saved.
        """
        with open(Path(path) / cls.vectorizer_config_file) as f:
            return json.load(f)

    def fit_transform(self, texts: Union[str, List[str]], *args, **kwargs) -> np.array:
        """
        Train vectorizer and transform texts.

        Args:
            texts (Union[str, List[str]]): The input texts or a list of texts to fit the vectorizer.

        Returns:
            np.array: An array of numerical vectors representing the input texts.
        """
        self.fit(texts)
        return self.transform(texts)

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
        return self.tokenizer.tokenize(inputs)


class GetTokenVectorsMixIn:
    @abstractmethod
    def get_token_vectors(self, texts: TokenizerInputs) -> EncodeTokensOutputs:
        """
        Tokenizes each input text and obtains a tuple list of (token, token_vector) for each input text.

        Args:
            texts (TokenizerInputs): The input text or a list of texts to tokenize.

        Returns:
            EncodeTokensOutputs: Each internal list consists of a tuple of tokenized words and their vector representations.

        Raises:
            NotImplementedError: This method must be implemented in derived classes.
        """
        raise NotImplementedError()
