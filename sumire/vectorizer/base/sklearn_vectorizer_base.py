import pickle
from abc import ABC
from pathlib import Path
from typing import List, Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sumire.tokenizer import TokenizerInputs
from sumire.vectorizer.base.common import BaseVectorizer


class SkLearnVectorizerBase(BaseVectorizer, ABC):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.vectorizer = None
        self.vectorizer: Union[CountVectorizer, TfidfVectorizer]

    def fit(self, texts: TokenizerInputs, *args, **kwargs) -> None:
        """
        Fits the vectorizer to the input texts.

        Args:
            texts (TokenizerInputs): The input texts to fit the vectorizer.

        Returns:
            None
        """
        tokenized = [" ".join(i) for i in self.tokenize(texts)]
        self.vectorizer.fit(tokenized)

    def transform(self, texts: TokenizerInputs, *args, **kwargs) -> np.array:
        """
        Transforms the input texts into numerical vectors.

        Args:
            texts (TokenizerInputs): The input texts to transform.

        Returns:
            np.array: An array of numerical vectors representing the input texts.
        """
        tokenized = [" ".join(i) for i in self.tokenize(texts)]
        return self.vectorizer.transform(tokenized).toarray()

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Saves the pretrained vectorizer and tokenizer to the specified path.

        Args:
            path (str or Path): The directory path where the pretrained vectorizer and tokenizer will be saved.

        Returns:
            None
        """
        self.tokenizer.save_pretrained(path)
        self._save_vectorizer_config(path, self.__class__.__name__)
        path = Path(path)
        with open(path / f"{self.__class__.__name__}.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        raise NotImplementedError()

    def decode(self, data: np.array) -> List[List[str]]:
        """
        Decodes the numerical vectors into tokenized texts.

        Args:
            data (np.array): The numerical vectors to decode.

        Returns:
            List[List[str]]: The decoded tokenized texts.
        """

        vocabs = self.vectorizer.get_feature_names_out()
        ret = []
        for row in (data > 0).tolist():
            decoded = []
            for col, vocab in zip(row, vocabs):  # TODO: use pandas pd.DataFrame(data, columns=vocabs) > 0
                if col:
                    decoded.append(vocab)
            ret.append(decoded)
        return ret
