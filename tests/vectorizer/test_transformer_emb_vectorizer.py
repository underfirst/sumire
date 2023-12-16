import pytest

from sumire.vectorizer.transformer_emb import TRANSFORMERS_MODEL_CARDS, ModelCard, TransformerEmbeddingVectorizer
from tests.vectorizer._test_get_token_vectors import _test_get_token_vectors
from tests.vectorizer._test_vectorizer import _test_vectorizer


def test_common():
    # testing different pooling type.
    for pooling_method in ["cls", "mean", "max"]:
        vectorizer = TransformerEmbeddingVectorizer(pooling_method=pooling_method)
        _test_vectorizer(vectorizer)
        _test_get_token_vectors(vectorizer)


@pytest.mark.parametrize("model_card", TRANSFORMERS_MODEL_CARDS)
def test_transformer_model_type(model_card: ModelCard):
    # testing different model type.
    vectorizer = TransformerEmbeddingVectorizer(pretrained_model_name_or_path=model_card.model_name)
    _test_vectorizer(vectorizer, head=50)
    _test_get_token_vectors(vectorizer)
