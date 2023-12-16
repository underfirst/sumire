from pathlib import Path
from typing import Dict, Optional, Type, Union

import inflection
from loguru import logger

from sumire.tokenizer.base import BaseTokenizer, TokenizerInputs, TokenizerOutputs, TokenizerType
from sumire.tokenizer.jumanpp import JumanppTokenizer
from sumire.tokenizer.mecab import MecabTokenizer
from sumire.tokenizer.spacy_ginza import SpacyGinzaTokenizer
from sumire.tokenizer.spm import SentencePieceTokenizer
from sumire.tokenizer.sudachi import SudachiTokenizer

TokenizerClass = Type[Union[MecabTokenizer,
JumanppTokenizer,
SpacyGinzaTokenizer,
SentencePieceTokenizer,
SudachiTokenizer,
MecabTokenizer]]

tokenizer_classes: Dict[str, TokenizerClass] = {}
for tokenizer_class in [MecabTokenizer, JumanppTokenizer, SpacyGinzaTokenizer, SentencePieceTokenizer, SudachiTokenizer,
                        MecabTokenizer]:
    tokenizer_classes[tokenizer_class.__name__] = tokenizer_class


def tokenizer_dict(name: str) -> TokenizerClass:
    class_name_lookup = {
        MecabTokenizer.__name__: ["mecab"],
        JumanppTokenizer.__name__: ["juman", "jumanpp"],
        SpacyGinzaTokenizer.__name__: ["spacy", "ginza", "spacyginza", "spacy_ginza"],
        SentencePieceTokenizer.__name__: ["sentencepiece", "sp", "spm"],
        SudachiTokenizer.__name__: ["sudachi", "sudachipy"]
    }
    for key, values in class_name_lookup.items():
        if inflection.underscore(name).lower() in values:
            return tokenizer_classes[key]
    raise ValueError(f"Tokenizer `{name}` is not implemented yet.")


class AutoJapaneseTokenizer(BaseTokenizer):
    """
        AutoJapaneseTokenizer automatically selects the tokenizer
        based on the given path or uses MecabTokenizer if no path is provided.

        Args:
            path (str, optional): The directory path to a saved tokenizer configuration.
                If not provided, a simple MecabTokenizer is used.

        Example:
            >>> tokenizer = AutoJapaneseTokenizer()
            >>> text = texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
            >>> tokens = tokenizer.tokenize(text)
            >>> tokens[0]
            ['これ', 'は', 'テスト', '文', 'です', '。']
        """
    def __init__(self, path: Optional[str] = None, *args, **kwargs):
        super().__init__()
        if path is None:
            logger.info("AutoJapaneseTokenizer with no initialization is same as simple MecabTokenizer().")
            self.tokenizer = MecabTokenizer()
        else:
            self.tokenizer = AutoJapaneseTokenizer.from_pretrained(path)

    def tokenize(self, inputs: TokenizerInputs) -> TokenizerOutputs:
        """
        Tokenizes input text.

        Args:
            inputs (str or List[str]): Text or list of texts to tokenize.

        Returns:
            List[List[str]]: List of tokenized texts.
        """
        return self.tokenizer.tokenize(inputs)

    def save_pretrained(self, path: Union[str, Path]):
        """
        Saves tokenizer configuration to the specified path.

        Args:
            path (str or Path): Directory path to save the configuration.
        """
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> TokenizerType:
        """
        Loads a tokenizer from a saved configuration.

        Args:
            path (str or Path): Directory path to the saved configuration.

        Returns:
            TokenizerType: Tokenizer instance initialized with the saved configuration.
        """
        try:
            tokenizer = tokenizer_dict(str(path))()
            return tokenizer
        except ValueError:
            pass
        configs = cls._load_config(path)
        class_name = configs["tokenizer_class"]
        logger.info("Loaded config", configs["tokenizer_class"], configs)
        return tokenizer_classes[class_name].from_pretrained(path)
