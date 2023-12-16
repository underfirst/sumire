from abc import ABC
from pathlib import Path
from typing import List, Union

from loguru import logger
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from sumire.vectorizer.base.common import BaseVectorizer, GetTokenVectorsMixIn


class TransformersVectorizerBase(BaseVectorizer, GetTokenVectorsMixIn, ABC):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer

    def fit(self, texts: Union[str, List[str]], *args, **kwargs) -> None:
        """
        This method is not implemented for TransformersVectorizerBase.

        Args:
            texts (Union[str, List[str]]): The input texts for fitting the vectorizer.

        Returns:
            None
        """
        logger.warning("This method is not implemented for TransformersVectorizerBase.")

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Saves the pretrained vectorizer, tokenizer, and model to the specified path.

        Args:
            path (str or Path): The directory path where the pretrained vectorizer, tokenizer, and model will be saved.

        Returns:
            None
        """
        self._save_vectorizer_config(path, self.__class__.__name__)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    @classmethod
    def load_init_args(cls, path: str):
        """
        Loads initialization arguments for the vectorizer from the specified path.

        Args:
            path (str): The directory path where the initialization arguments are stored.

        Returns:
            dict: A dictionary containing the initialization arguments.

        Raises:
            ValueError: If the vectorizer configuration file does not exist at the specified path.
        """
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path)
        configs = cls._load_vectorizer_config(path)
        init_args = configs["init_args"]
        init_args["tokenizer"] = tokenizer
        init_args["model"] = model
        return init_args
