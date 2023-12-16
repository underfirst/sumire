from os import environ, makedirs
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Union

import ipadic
import unidic
import unidic_lite
from loguru import logger
from MeCab import Tagger
from unidic.download import download_version

from sumire.tokenizer.base import BaseTokenizer, TokenizerInputs, TokenizerOutputs
from sumire.utils.download_file import download_file
from sumire.utils.run_sh import run_sh

AvailableDictionaries = Literal["unidic", "unidic-lite", "ipadic", "mecab-ipadic-neologd", "mecab-unidic-neologd"]


class MecabTokenizer(BaseTokenizer):
    """
    Tokenizer class using MeCab for Japanese text tokenization.

    Args:
        dictionary (AvailableDictionaries, optional): Dictionary to use for tokenization. Default is "unidic".

    Example:
        >>> tokenizer = MecabTokenizer()
        >>> text = texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
        >>> tokens = tokenizer.tokenize(text)
        >>> tokens
        [['これ', 'は', 'テスト', '文', 'です', '。'], ['別', 'の', 'テキスト', 'も', 'トークン', '化', 'し', 'ます', '。']]
    """
    def __init__(self, dictionary: AvailableDictionaries = "unidic", *args, **kwargs):
        super().__init__()
        self.init_args["dictionary"] = dictionary

        self.dictionary = dictionary
        self.tagger = self.setup_tagger(dictionary)

    def setup_tagger(self, dictionary: AvailableDictionaries) -> Tagger:
        """
        Sets up the MeCab tagger based on the selected dictionary.

        Args:
            dictionary (AvailableDictionaries): Dictionary to use for tokenization.

        Returns:
            Tagger: MeCab tagger configured with the selected dictionary.
        """
        mecab_args = self._install_mecab_if_not_installed()
        if dictionary == "unidic":
            if not (Path(unidic.DICDIR) / "mecabrc").exists():
                download_version()
            return Tagger(mecab_args + f" -Owakati -d '{unidic.DICDIR}'")
        elif dictionary == "unidic-lite":
            return Tagger(mecab_args + f" -Owakati -d '{unidic_lite.DICDIR}'")
        elif dictionary == "ipadic":
            return Tagger(mecab_args + f" -Owakati -d '{ipadic.DICDIR}'")
        elif dictionary == "mecab-ipadic-neologd":
            self._install_mecab_ipadic_neologd()
            return Tagger(mecab_args + f" -Owakati -d '{environ['HOME']}/.local/sumire/mecab-ipadic-neologd'")
        elif dictionary == "mecab-unidic-neologd":
            self._install_mecab_unidic_neologd()
            return Tagger(mecab_args + f" -Owakati -d '{environ['HOME']}/.local/sumire/mecab-unidic-neologd'")

    def _install_mecab_if_not_installed(self) -> str:
        """
        Installs MeCab if it is not already installed.

        Returns:
            str: MeCab command-line arguments to detect MeCab execution binary.
        """
        if run_sh("which mecab") == "":  # TODO: macだとwhichに依存性があるかもしれない
            if run_sh(f"ls {environ['HOME']}/.local/sumire/mecab") != "":
                return  f"-r {environ['HOME']}/.local/sumire/mecab"
            logger.warning(
                    "No mecab detected.\n"
                    "Install mecab to `$HOME/.local/sumire/mecab.\n"
                    "It may take a while...")
            with TemporaryDirectory() as dir_path:
                dir_path = Path(dir_path)
                download_file(
                    "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE",
                    dir_path / "mecab.tar.gz",
                )
                makedirs(f"{environ['HOME']}/.local/sumire/", exist_ok=True)
                run_sh(f"""cd {dir_path};
                tar -xvf {dir_path / 'mecab.tar.gz'};
                cd mecab-0.996;
                ./configure --enable-utf8-only --prefix=$HOME/.local/sumire/mecab;
                make install""")
                return f"-r {environ['HOME']}/.local/sumire/mecab"
        else:
            return ""

    def _install_mecab_ipadic_neologd(self) -> None:
        """
        Installs MeCab-IPADic-NEologd dictionary if it is not already installed.
        It may require git login.
        """
        if not Path(f"{environ['HOME']}/.local/sumire/mecab-ipadic-neologd").exists():
            logger.warning(
                "No mecab-ipadic-neologd detected to sumire cache directory.\n"
                "Install mecab-ipadic-neologd to $HOME/.local/sumire/mecab-ipadic-neologd.\n"
                "It may take a while..."
            )
            with TemporaryDirectory() as dir_path:
                dir_path = Path(dir_path)
                makedirs(f"{environ['HOME']}/.local/sumire/", exist_ok=True)
                run_sh(f"""cd {dir_path};
                git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git;
                cd mecab-ipadic-neologd;
                PATH=$HOME/.local/sumire/mecab/bin:$PATH ./bin/install-mecab-ipadic-neologd -y -p {environ['HOME']}/.local/sumire/mecab-ipadic-neologd;
                """)

    def _install_mecab_unidic_neologd(self) -> None:
        """
        Installs MeCab-UniDic-NEologd dictionary if it is not already installed.
        It may require git login.
        """
        if not Path(f"{environ['HOME']}/.local/sumire/mecab-unidic-neologd").exists():
            logger.warning(
                "No mecab-ipadic-neologd detected to sumire cache directory.\n"
                "Install mecab-ipadic-neologd to $HOME/.local/sumire/mecab-unidic-neologd.\n"
                "It may take a while...")
            with TemporaryDirectory() as dir_path:
                dir_path = Path(dir_path)
                makedirs(f"{environ['HOME']}/.local/sumire/", exist_ok=True)
                run_sh(
                    f"""cd {dir_path};
                git clone --depth 1 https://github.com/neologd/mecab-undic-neologd.git;
                cd mecab-ipadic-neologd;
                PATH=$HOME/.local/sumire/mecab/bin:$PATH ./bin/install-mecab-unidic-neologd -y -p {environ['HOME']}/.local/sumire/mecab-unidic-neologd;
                """
                )

    def tokenize(self, inputs: TokenizerInputs) -> TokenizerOutputs:
        """
        Tokenizes input text using MeCab.

        Args:
            inputs (str or List[str]): Text or list of texts to tokenize.

        Returns:
            TokenizerOutputs: Tokenized texts as lists of strings.

        Example:
            >>> tokenizer = MecabTokenizer()
            >>> texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
            >>> tokens = tokenizer.tokenize(texts)
            >>> tokens
            [['これ', 'は', 'テスト', '文', 'です', '。'], ['別', 'の', 'テキスト', 'も', 'トークン', '化', 'し', 'ます', '。']]
        """
        ret = []
        if isinstance(inputs, str):
            inputs = [inputs]
        for text in inputs:
            ret.append(self.tagger.parse(text).split())
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
            MecabTokenizer: A tokenizer loaded from the specified configuration.
        """
        configs = cls._load_config(path)
        return MecabTokenizer(**configs["init_args"])
