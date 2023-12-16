from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger

from sumire.tokenizer import BaseTokenizer, MecabTokenizer, JumanppTokenizer, SpacyGinzaTokenizer, SudachiTokenizer, \
    AutoJapaneseTokenizer, SentencePieceTokenizer

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data/test/test.txt"

test_lines = [line for line in open(TEST_DATA_DIR).readlines()]


def _test_tokenizer(tokenizer: BaseTokenizer, head: int = -1):
    with TemporaryDirectory() as temp_dir:
        logger.info(f"start {tokenizer.__class__.__name__}")

        tokenizer.fit(test_lines)  # for sentencepiece.
        tokenizer.save_pretrained(temp_dir)
        loaded_tokenizer = AutoJapaneseTokenizer.from_pretrained(temp_dir)

        single_ret = tokenizer.tokenize(test_lines[0])

        assert isinstance(single_ret[0], list)
        assert isinstance(single_ret[0][0], str)
        assert len(single_ret) == 1

        ret = tokenizer.tokenize(test_lines[:head])
        loaded_ret = loaded_tokenizer.tokenize(test_lines[:head])
        assert isinstance(ret[0], list)
        assert isinstance(ret[0][0], str)
        assert len(ret) == len(test_lines[:head])
        assert ret == loaded_ret, "from_pretrained() fail."


def test_mecab_unidic_tokenizer():
    _test_tokenizer(MecabTokenizer())


def test_mecab_unidic_lite_tokenizer():
    _test_tokenizer(MecabTokenizer("unidic-lite"))


def test_mecab_ipadic_tokenizer():
    _test_tokenizer(MecabTokenizer("ipadic"))


# def test_mecab_ipadic_neologd_tokenizer():
#     _test_tokenizer(MecabTokenizer("mecab-ipadic-neologd"))
#
#
# def test_mecab_unidic_neologd_tokenizer():
#     _test_tokenizer(MecabTokenizer("mecab-unidic-neologd"))
#

def test_jumanpp_tokenizer():
    _test_tokenizer(JumanppTokenizer(), head=10)


def test_spacy_ginza_tokenizer():
    _test_tokenizer(SpacyGinzaTokenizer(), head=10)


def test_sudachi_tokenizer():
    _test_tokenizer(SudachiTokenizer())


def test_spm_tokenizer():
    _test_tokenizer(SentencePieceTokenizer(vocab_size=15000))
