from sumire.tokenizer import MecabTokenizer
from sumire.vectorizer.count import CountVectorizer
from tests.vectorizer._test_vectorizer import _test_vectorizer
from tests.vectorizer._test_vectorizer_decode import _test_vectorizer_decode


def test_count_vectorizer():
    vectorizer = CountVectorizer(MecabTokenizer(), max_features=1000)
    _test_vectorizer(vectorizer)
    _test_vectorizer_decode(vectorizer)
