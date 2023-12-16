from multiprocessing import cpu_count
from os import environ, makedirs
from pathlib import Path
from typing import Union

from loguru import logger
from pyknp import Juman

from sumire.tokenizer.base import BaseTokenizer, TokenizerInputs, TokenizerOutputs
from sumire.utils.download_file import download_file
from sumire.utils.run_sh import run_sh


class JumanppTokenizer(BaseTokenizer):
    """
    Tokenizer class using Juman++ for Japanese text tokenization.

     Example:
        >>> tokenizer = JumanppTokenizer()
        >>> text = "これはテスト文です。"
        >>> tokens = tokenizer.tokenize(text)
        >>> tokens
        [['これ', 'は', 'テスト', '文', 'です', '。']]
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = self._install_jumanpp_if_not_installed()

    def _install_jumanpp_if_not_installed(self) -> Juman:
        """
        Installs Juman++ if it is not already installed and returns a Juman instance.

        Returns:
            Juman: Juman instance for tokenization.

        Raises:
            EnvironmentError: If Juman++ cannot be installed automatically.
        """
        jumanpp_path = run_sh("which jumanpp")  # TODO: whichは依存性があるかもしれないのでかえたい.

        if jumanpp_path:
            return Juman()

        if jumanpp_path == "":
            sumire_jumanpp_path = run_sh("which $HOME/.local/sumire/jumanpp/bin/jumanpp")
            if sumire_jumanpp_path != "":
                return Juman(f"{environ['HOME']}/.local/sumire/jumanpp/bin/jumanpp")
            logger.warning(
                "No jumanpp detected.\n"
                "Install Jumanpp to `$HOME/.local/sumire/jumanpp`.\n"
                "It may take a while..."
            )
            makedirs(f"{environ['HOME']}/.local/sumire", exist_ok=True)
            dir_path = Path(f"{environ['HOME']}/.local/sumire")
            download_file(
                "https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc4/jumanpp-2.0.0-rc4.tar.xz",
                dir_path / "jumanpp-2.0.0-rc4.tar.xz",
            )
            cmd = f"""cd {dir_path};
                       tar -xvf jumanpp-2.0.0-rc4.tar.xz;
                       cd jumanpp-2.0.0-rc4;
                       mkdir -p build_dir;
                       cd build_dir;
                       cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local/sumire/jumanpp;
                       make install -j {cpu_count()}"""

            logger.info(f"Installing jumanpp with following command.\n{cmd}")
            ret = run_sh(cmd, True)
            logger.info(ret)
            logger.info("Execution done with following stdout.")
            if run_sh("which $HOME/.local/sumire/jumanpp/bin/jumanpp") != "":
                return Juman(f"{environ['HOME']}/.local/sumire/jumanpp/bin/jumanpp")

        raise OSError("Cannot install jumanpp automatically.")

    def tokenize(self, inputs: TokenizerInputs) -> TokenizerOutputs:
        """
        Tokenizes input text.

        Args:
            inputs (str or List[str]): Text or list of texts to tokenize.

        Returns:
            List[List[str]]: List of tokenized texts.

        Example:
            >>> tokenizer = JumanppTokenizer()
            >>> text = "これはテスト文です。"
            >>> tokens = tokenizer.tokenize(text)
            >>> tokens
            [['これ', 'は', 'テスト', '文', 'です', '。']]
        """

        if isinstance(inputs, str):
            inputs = [inputs]
        ret = []
        for text in inputs:
            ret.append([i.midasi for i in self.tokenizer.analysis(text)])
        return ret

    def save_pretrained(self, path: Union[str, Path]):
        """
        Saves tokenizer configuration to the specified path.

        Args:
            path (str or Path): Directory path to save the configuration.
        """
        self._save_pretrained(path, self.__class__.__name__)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Loads a tokenizer from a saved configuration.

        Args:
            path (str or Path): Directory path to the saved configuration.

        Returns:
            JumanppTokenizer: Tokenizer instance initialized with the saved configuration.
        """
        configs = cls._load_config(path)
        return JumanppTokenizer(**configs["init_args"])


if __name__ == '__main__':
    JumanppTokenizer()
