from sumire.vectorizer.swem import W2VSWEMVectorizer
from tests.vectorizer._test_get_token_vectors import _test_get_token_vectors
from tests.vectorizer._test_vectorizer import _test_vectorizer


def test_common_parameter():
    for pooling_method in ["mean", "max"]:
        vectorizer = W2VSWEMVectorizer(pooling_method=pooling_method)
        _test_vectorizer(vectorizer, head=100)
        _test_get_token_vectors(vectorizer)
