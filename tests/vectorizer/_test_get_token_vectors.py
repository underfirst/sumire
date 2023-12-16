from typing import Union

import numpy as np

from sumire.vectorizer.swem import W2VSWEMVectorizer
from sumire.vectorizer.transformer_emb import TransformerEmbeddingVectorizer
from tests.vectorizer._test_vectorizer import test_lines


def _test_get_token_vectors(vectorizer: Union[TransformerEmbeddingVectorizer, W2VSWEMVectorizer]):
    vectorized = vectorizer.get_token_vectors(test_lines[:5])
    assert len(vectorized) == 5
    token_vectorizer_list = vectorized[0]
    token_vectorizer = token_vectorizer_list[0]
    token, token_vector = token_vectorizer
    assert isinstance(token, str)
    assert isinstance(token_vector, np.ndarray)
