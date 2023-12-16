from sumire.vectorizer.base.sklearn_vectorizer_base import SkLearnVectorizerBase
from tests.vectorizer._test_vectorizer import test_lines


def _test_vectorizer_decode(vectorizer: SkLearnVectorizerBase):
    ret = vectorizer.fit_transform(test_lines)
    decoded = vectorizer.decode(ret)

    assert isinstance(decoded, list)
    assert len(decoded) == ret.shape[0]
