import json
from os import environ, remove
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from urllib.parse import urlsplit

import gensim
import numpy as np
from loguru import logger
from pydantic import BaseModel

from sumire.tokenizer import BaseTokenizer, MecabTokenizer, SudachiTokenizer
from sumire.utils.download_file import download_file
from sumire.utils.unbz_file import unbz_file
from sumire.utils.untar_file import untar_file
from sumire.utils.unzip_file import unzip_file
from sumire.vectorizer.base.common import BaseVectorizer


class ModelCard(BaseModel):
    name: str
    url: str
    tokenizer_name: str = "mecab"
    description: str = ""


COMMON_MODELS = []
for file in (Path(__file__).parent.parent / "resources" / "model_card" / "gensim").glob("**/*.json"):
    with open(file) as f:
        data = ModelCard(**json.load(f))
        COMMON_MODELS.append(data)

COMMON_MODEL_NAMES = [card.name for card in COMMON_MODELS]


class W2VSWEMVectorizer(BaseVectorizer):
    """
    W2VSWEMVectorizer is a vectorizer class that uses Word2Vec-based embeddings for text data.

    To use chive model, give model_name_or_path to
    "chive-{version}-mc{min count}" such as `W2VSWEMVectorizer("chive-1.0-mc5")`.
    The model alias name are the "name" key in "sumire/resource/model_card/gensim/chive/*.json".

    To use cl-tohoku japanese wikipedia entity vectors,
    give model_name_or_path to "{releas date}/jawiki.{all|entity|word}_vectors.{dimension}d"
    such as `W2VSWEMVectorizer("20180402/jawiki.entity_vectors.100d.json")`
    The model alias name are the "name" key of
    "sumire/resource/model_card/gensim/cl-tohoku_jawiki_vector/**/*.json".


    Args:
        model_name_or_path (str, optional): The model name or path to Word2Vec embeddings.
            Default is "20190520/jawiki.word_vectors.100d".
                The alias names are in name key of `resources/model_card/gensim/**/*.json`
        pooling_method (str, optional): The pooling method for
            aggregating word vectors ("mean" or "max"). Default is "mean".
        download_timeout (int, optional): The timeout for downloading embeddings.
            Default is 3600.
        tokenizer (BaseTokenizer, optional): The tokenizer to use.
            If not provided, a default MecabTokenizer is used.

    Attributes:
        w2v_dir (Path): The directory for storing Word2Vec embeddings.

    Examples:
        >>> from sumire.vectorizer.swem import W2VSWEMVectorizer
        >>> vectorizer = W2VSWEMVectorizer()
        >>> texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
        >>> vectors = vectorizer.transform(texts)
        >>> vectors.shape
        (2, 100)
    """
    w2v_dir = Path(environ["HOME"]) / ".local/sumire/gensim"

    def __init__(self,
                 model_name_or_path: str = "20190520/jawiki.word_vectors.100d",
                 pooling_method: Literal["mean", "max"] = "mean",
                 download_timeout: int = 3600,
                 tokenizer: Optional[BaseTokenizer] = None):
        super().__init__()
        self.init_args["model_name_or_path"] = model_name_or_path
        self.init_args["pooling_method"] = pooling_method
        self.init_args["download_timeout"] = download_timeout
        self.model_name_or_path = model_name_or_path
        self.pooling = pooling_method
        self.download_timeout = download_timeout

        self.tokenizer = MecabTokenizer()

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name_or_path in COMMON_MODEL_NAMES:
            common_model = COMMON_MODELS[COMMON_MODEL_NAMES.index(model_name_or_path)]
            if common_model.tokenizer_name == "mecab":
                self.tokenizer = MecabTokenizer()
            elif common_model.tokenizer_name == "mecab-ipadic-neologd":
                self.tokenizer = MecabTokenizer("mecab-ipadic-neologd")
            elif common_model.tokenizer_name == "sudachi":
                self.tokenizer = SudachiTokenizer(normalize=True)
            model_name_or_path = COMMON_MODELS[COMMON_MODEL_NAMES.index(model_name_or_path)].url
        w2v_bin_path = self.setup_w2v_if_not_installed(model_name_or_path)
        if model_name_or_path.find("chive") != -1:
            self.model = gensim.models.KeyedVectors.load(str(w2v_bin_path))
        else:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(str(w2v_bin_path),
                                                                         binary=False)

    def setup_w2v_if_not_installed(self, path_or_url: str) -> Path:
        """
        Set up Word2Vec embeddings if they are not already installed.

        Args:
            path_or_url (str): The path or URL to Word2Vec embeddings.

        Returns:
            Path: The path to the Word2Vec binary file.

        """
        if Path(path_or_url).exists():
            return Path(path_or_url)

        file_path = Path(str(self.w2v_dir) + urlsplit(path_or_url).path)
        w2v_binary_file_path = file_path

        if file_path.name.endswith(".zip"):
            dir_path = file_path.parent / file_path.name.replace(".zip", "")
            num_bin_file = 0
            for file in dir_path.glob("*"):
                if file.name.find(".bin") != -1:
                    w2v_binary_file_path = file
                    num_bin_file += 1
            if num_bin_file != 1:
                logger.warning("gensim file may not automatically installed.")
        elif file_path.name.endswith(".bz2"):
            w2v_binary_file_path = file_path.parent / file_path.name.replace(".bz2", "")
        elif file_path.name.find("chive") != -1:
            chive_extract_path = file_path.parent / file_path.name.replace(".tar.gz", "")
            if not chive_extract_path.exists() and file_path.exists() and file_path.name.endswith(".tar.gz"):
                untar_file(file_path, chive_extract_path)
            for file in chive_extract_path.glob("**/*"):
                if file.name.endswith(".kv"):
                    w2v_binary_file_path = file

        if w2v_binary_file_path.exists():
            return w2v_binary_file_path

        logger.info(f"gensim file {path_or_url} download to {file_path}. It may take a while...")
        download_file(path_or_url, file_path, timeout=self.download_timeout)

        if file_path.name.endswith(".zip"):
            dir_path = file_path.parent / file_path.name.replace(".zip", "")
            if file_path.name.endswith(".zip"):
                unzip_file(file_path, dir_path)
            remove(file_path)
        elif file_path.name.endswith(".bz2"):
            unbz_file(file_path, w2v_binary_file_path)
            remove(file_path)
        elif file_path.name.find("chive") != -1:
            chive_extract_path = file_path.parent / file_path.name.replace(".tar.gz", "")
            if not chive_extract_path.exists():
                untar_file(file_path, chive_extract_path)
            for file in chive_extract_path.glob("**/*"):
                if file.name.endswith(".kv"):
                    w2v_binary_file_path = file
            remove(file_path)
        return w2v_binary_file_path

    def fit(self, texts: Union[str, List[str]], *args, **kwargs) -> None:
        """
        Fit the vectorizer (not implemented).

        Args:
            texts (str or List[str]): Input texts for fitting the vectorizer.

        Returns:
            None

        """
        pass

    def transform(self, texts: Union[str, List[str]], *args, **kwargs) -> np.array:
        """
        Transform input texts into word vectors.

        Args:
            texts (str or List[str]): Input texts to be transformed.

        Returns:
            np.array: Transformed text vectors.

        """
        tokenized = self.tokenize(texts)
        ret = []
        for tokens in tokenized:
            if self.pooling == "mean":
                try:
                    vec = self.model.get_mean_vector(tokens)
                except (KeyError, ValueError):
                    vec = np.zeros(self.model.vector_size)
                ret.append(vec)
            elif self.pooling == "max":
                token_vectors = [np.zeros(self.model.vector_size)]
                for token in tokens:
                    try:
                        token_vector = self.model.get_vector(token)
                        token_vectors.append(token_vector)
                    except KeyError:
                        continue
                vec = np.max(np.concatenate(token_vectors), axis=0)
                ret.append(vec)
        return np.stack(ret)

    def get_token_vectors(self, texts: Union[str, List[str]]) -> List[List[Tuple[str, np.array]]]:
        """
        Tokenizes each input text and obtains a tuple list of (token, token_vector) for each input text.

        Args:
            texts (TokenizerInputs): The input text or a list of texts to tokenize.

        Returns:
            EncodeTokensOutputs: Each internal list consists of
                a tuple of tokenized words and their vector representations.


        Example:
        >>> from sumire.vectorizer.swem import W2VSWEMVectorizer
        >>> vectorizer = W2VSWEMVectorizer()
        >>> texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
        >>> vectors = vectorizer.get_token_vectors(texts)
        >>> len(vectors)
        2
        >>> isinstance(vectors[0][0][0], str)
        True
        >>> vectors[0][0][1].shape == (100, )
        True
        """
        ret = []
        for tokens in self.tokenize(texts):
            part = []
            for token in tokens:
                try:
                    word_vector = self.model.get_vector(token)
                    part.append((token, word_vector))
                except KeyError:
                    continue
            ret.append(part)
        return ret

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """
        Save the vectorizer and tokenizer to a specified path.

        Args:
            path (str or Path): The directory path to save the vectorizer.

        Returns:
            None

        """
        self.tokenizer.save_pretrained(path)
        self._save_vectorizer_config(path, self.__class__.__name__)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Load a pretrained vectorizer from a specified path.

        Args:
            path (str or Path): The directory path to load the pretrained vectorizer from.

        Returns:
            W2VSWEMVectorizer: The loaded pretrained vectorizer.
        """
        config = cls._load_vectorizer_config(path)
        return cls(**config["init_args"])
