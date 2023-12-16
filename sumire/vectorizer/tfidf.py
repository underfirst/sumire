import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

from sklearn.feature_extraction import text

from sumire.tokenizer import AutoJapaneseTokenizer, BaseTokenizer, TokenizerType
from sumire.vectorizer.base.sklearn_vectorizer_base import SkLearnVectorizerBase


class TfidfVectorizer(SkLearnVectorizerBase):
    """
    TfidfVectorizer is a vectorizer class that uses TfIdf implemented in scikit-learn.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        lowercase (bool, optional): Whether to convert all characters to lowercase before tokenization.
            Defaults to True.
        stop_words (str, List[str], or None, optional): The stop words to use for filtering tokens.
            Defaults to None.
        ngram_range (tuple, optional): The range of n-grams to extract as features.
            Defaults to (1, 1) (i.e., only unigrams).
        max_df (float or int, optional): The maximum document frequency for
            a token to be included in the vocabulary.
            Can be a float in the range [0.0, 1.0] or an integer. Defaults to 1.0 (i.e., no filtering).
        min_df (float or int, optional): The minimum document frequency for
            a token to be included in the vocabulary.
            Can be a float in the range [0.0, 1.0] or an integer. Defaults to 1 (i.e., no filtering).
        max_features (int or None, optional): The maximum number of features (tokens)
            to include in the vocabulary.
            Defaults to None (i.e., no limit).
        norm (str, optional): The normalization method for tf-idf vectors. Defaults to "l2".
        use_idf (bool, optional): Whether to use inverse document frequency in tf-idf computation. Defaults to True.
        smooth_idf (bool, optional): Whether to smooth idf weights. Defaults to True.
            sublinear_tf (bool, optional): Whether to apply sublinear tf scaling. Defaults to False.

    Returns:
        None

    Example:
        >>> from sumire.tokenizer import MecabTokenizer
        >>> texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
        >>> tokenizer = MecabTokenizer()
        >>> vectorizer = TfidfVectorizer(tokenizer=tokenizer)
        >>> vectorizer.fit(texts)
        >>> transformed = vectorizer.transform(texts)
    """

    def __init__(
            self,
            tokenizer: Union[str, TokenizerType] = "mecab",
            lowercase: bool = True,
            stop_words: Optional[List[str]] = None,
            ngram_range: Tuple = (1, 1),
            max_df: float = 1.0,
            min_df: int = 1,
            max_features: Optional[int] = None,
            norm: str = "l2",
            use_idf: bool = True,
            smooth_idf: bool = True,
            sublinear_tf: bool = False,
            *args, **kwargs,
    ):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = AutoJapaneseTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, BaseTokenizer):
            pass
        self.tokenizer = tokenizer
        self.init_args["lowercase"] = lowercase
        self.init_args["stop_words"] = stop_words
        self.init_args["ngram_range"] = ngram_range
        self.init_args["max_df"] = max_df
        self.init_args["min_df"] = min_df
        self.init_args["max_features"] = max_features
        self.init_args["norm"] = norm
        self.init_args["use_idf"] = use_idf
        self.init_args["smooth_idf"] = smooth_idf
        self.init_args["sublinear_tf"] = sublinear_tf
        self.vectorizer = text.TfidfVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Loads a pretrained TfidfVectorizer from the specified path.

        Args:
            path (Union[str, Path]): The directory path where the pretrained TfidfVectorizer is saved.

        Returns:
            TfidfVectorizer: A TfidfVectorizer instance loaded with the pretrained model and configuration.

        Example:
            >>> pretrained_path = "/path/to/pretrained_model"
            >>> vectorizer = TfidfVectorizer.from_pretrained(pretrained_path)
        """
        config = cls._load_vectorizer_config(path)
        tokenizer = AutoJapaneseTokenizer.from_pretrained(path)
        init_args = config["init_args"]
        init_args["tokenizer"] = tokenizer

        obj = TfidfVectorizer(**init_args)
        path = Path(path)
        with open(path / f"{cls.__name__}.pkl", "rb") as f:
            vectorizer = pickle.load(f)
            obj.vectorizer = vectorizer
        return obj
