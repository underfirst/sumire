from pathlib import Path
from typing import Union

import spacy

from sumire.tokenizer.base import BaseTokenizer, TokenizerInputs, TokenizerOutputs


class SpacyGinzaTokenizer(BaseTokenizer):
    """
    Tokenizer class using SpaCy with the Ginza model for Japanese text tokenization.

    Example:
        >>> tokenizer = SpacyGinzaTokenizer()
        >>> text = "これはテスト文です。"
        >>> tokens = tokenizer.tokenize(text)
        >>> tokens
        [['これ', 'は', 'テスト', '文', 'です', '。']]
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = spacy.load("ja_ginza")

    def tokenize(self, inputs: TokenizerInputs) -> TokenizerOutputs:
        """
        Tokenizes input text using SpaCy with the Ginza model.

        Args:
            inputs (str or List[str]): Text or list of texts to tokenize.

        Returns:
            List[List[str]]: List of tokenized texts.

        Example:
            >>> tokenizer = SpacyGinzaTokenizer()
            >>> text = "これはテスト文です。"
            >>> tokens = tokenizer.tokenize(text)
            >>> tokens
            [['これ', 'は', 'テスト', '文', 'です', '。']]
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        ret = []
        for text in inputs:
            ret.append([token.text for token in self.tokenizer(text)])
        return ret

    def save_pretrained(self, path: Union[str, Path]):
        """
        Saves tokenizer configuration to the specified path.

        Args:
            path (str or Path): Directory path to save the configuration.
        """
        self._save_pretrained(path, tokenizer_class=self.__class__.__name__)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Loads a tokenizer from a saved configuration.

         Args:
             path (str or Path): Directory path to the saved configuration.

        Returns:
            SpacyGinzaTokenizer: Tokenizer instance initialized with the saved configuration.
        """
        configs = cls._load_config(path)
        return SpacyGinzaTokenizer(**configs["init_args"])
