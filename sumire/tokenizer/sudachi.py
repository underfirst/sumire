from pathlib import Path
from typing import Union

import sudachipy

from sumire.tokenizer.base import BaseTokenizer, TokenizerInputs, TokenizerOutputs


class SudachiTokenizer(BaseTokenizer):
    """
    Class for a custom tokenizer using SudachiPy.

    Args:
        dict_type (str, optional): Type of SudachiPy dictionary. Default is "full".
        normalize (bool, optional): Flag to normalize tokens. Default is False.

    Attributes:
        dict_type (str): Type of SudachiPy dictionary.
        normalize (bool): Flag to normalize tokens.
        tokenizer (sudachipy.Dictionary): SudachiPy tokenizer object.

    Example:
        >>> tokenizer = SudachiTokenizer()
        >>> text = texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
        >>> tokens = tokenizer.tokenize(text)
        >>> tokens
        [['これ', 'は', 'テスト', '文', 'です', '。'], ['別', 'の', 'テキスト', 'も', 'トークン', '化', 'し', 'ます', '。']]
    """

    def __init__(self, dict_type: str = "full", normalize: bool = False, *args, **kwargs):
        super().__init__()
        self.init_args["dict_type"] = dict_type
        self.init_args["normalize"] = normalize
        self.normalize = normalize
        self.dict_type = dict_type
        sudachi_dict = sudachipy.Dictionary(dict_type=dict_type)
        self.tokenizer = sudachi_dict.create()

    def tokenize(self, inputs: TokenizerInputs) -> TokenizerOutputs:
        """
        Tokenizes input text.

        Args:
            inputs (str or List[str]): Text or list of texts to tokenize.

        Returns:
            List[List[str]]: List of tokenized texts.

        Example:
            >>> tokenizer = SudachiTokenizer()
            >>> text = texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
            >>> tokens = tokenizer.tokenize(text)
            >>> tokens
            [['これ', 'は', 'テスト', '文', 'です', '。'], ['別', 'の', 'テキスト', 'も', 'トークン', '化', 'し', 'ます', '。']]
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        ret = []
        for text in inputs:
            items = []
            for tok in self.tokenizer.tokenize(text):
                if self.normalize:
                    items.append(tok.normalized_form())
                else:
                    items.append(tok.surface())
            ret.append(items)
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
        Loads tokenizer from a saved configuration.

        Args:
            path (str or Path): Directory path to the saved configuration.

        Returns:
            SudachiTokenizer: Tokenizer instance initialized with the saved configuration.
        """
        configs = cls._load_config(path)
        return SudachiTokenizer(**configs["init_args"])
