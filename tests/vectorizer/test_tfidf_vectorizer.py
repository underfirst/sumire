from sumire.tokenizer import MecabTokenizer
from sumire.vectorizer.tfidf import TfidfVectorizer
from tests.vectorizer._test_vectorizer import _test_vectorizer
from tests.vectorizer._test_vectorizer_decode import _test_vectorizer_decode


def test_tfidf_vectorizer():
    vectorizer = TfidfVectorizer(MecabTokenizer(), max_features=1000)
    _test_vectorizer(vectorizer)
    _test_vectorizer_decode(vectorizer)
