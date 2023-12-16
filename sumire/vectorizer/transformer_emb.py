import json
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from sumire.vectorizer.base.common import EncodeTokensOutputs
from sumire.vectorizer.base.transformer_vectorizer_base import TransformersVectorizerBase


class ModelCard(BaseModel):
    model_name: str
    description: str = ""


TRANSFORMERS_MODEL_CARDS = []
for json_file in (Path(__file__).parent.parent / "resources/model_card/transformers").glob("**/*.json"):
    with open(json_file) as f:
        TRANSFORMERS_MODEL_CARDS.append(ModelCard(**json.load(f)))


class TransformerEmbeddingVectorizer(TransformersVectorizerBase):
    """
    TransformerEmbeddingVectorizer is a vectorizer class that uses
        transformer-based embeddings (e.g., BERT) for text data.

    Tested model infomations are in /sumire/resources/model_card/transformers.

    Args:
        pretrained_model_name_or_path (str, optional): The pretrained model name or path.
            Default is "cl-tohoku/bert-base-japanese-v3".
        pooling_method (str, optional): The pooling method for
            aggregating embeddings ("cls", "mean", "max"). Default is "cls".
        batch_size (int, optional): The batch size for processing texts. Default is 32.
        max_length (int, optional): The maximum length of input sequences.
            If not provided, it is determined by the model's configuration.
        model (PreTrainedModel, optional): A pretrained transformer model.
            If not provided, it is loaded from the specified model name or path.
        tokenizer (PreTrainedTokenizer, optional): A pretrained tokenizer.
            If not provided, it is loaded from the specified model name or path.

    Examples:
        >>> from sumire.vectorizer.transformer_emb import TransformerEmbeddingVectorizer
        >>> vectorizer = TransformerEmbeddingVectorizer()
        >>> texts = ["This is a sample sentence.", "Another example."]
        >>> vectors = vectorizer.transform(texts)
        >>> vectors.shape
        (2, 768)  # Assuming a BERT model with 768-dimensional embeddings

    """

    def __init__(
            self,
            pretrained_model_name_or_path: str = "cl-tohoku/bert-base-japanese-v3",
            pooling_method: Literal["cls", "mean", "max"] = "cls",
            batch_size: int = 32,
            max_length: Optional[int] = None,
            model: Optional[PreTrainedModel] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Initialize a TransformerEmbeddingVectorizer.

        Args:
            pretrained_model_name_or_path (str, optional): The pretrained model name or path.
                Default is "cl-tohoku/bert-base-japanese-v3".
            pooling_method (str, optional): The pooling method for
                aggregating embeddings ("cls", "mean", "max"). Default is "cls".
            batch_size (int, optional): The batch size for processing texts.
                Default is 32.
            max_length (int, optional): The maximum length of input sequences.
                If not provided, it is determined by the model's configuration.
            model (PreTrainedModel, optional): A pretrained transformer model.
                If not provided, it is loaded from the specified model name or path.
            tokenizer (PreTrainedTokenizer, optional): A pretrained tokenizer.
                If not provided, it is loaded from the specified model name or path.

        """
        super().__init__()
        self.init_args["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        self.init_args["pooling_method"] = pooling_method
        self.init_args["max_length"] = max_length
        self.init_args["batch_size"] = batch_size
        self.pooling_method = pooling_method
        self.max_length = max_length

        if model is None or tokenizer is None:
            model = AutoModel.from_pretrained(pretrained_model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = tokenizer
        self.model = model
        if max_length is None:
            self.max_length = self.model.config.max_length
        self.batch_size = batch_size

    def transform(self, texts: Union[str, List[str]],
                  *args, **kwargs) -> np.array:
        """
        Transform input texts into transformer-based embeddings.

        Args:
            texts (str or List[str]): Input texts to be transformed.
            batch_size (int, optional): The batch size for processing texts. Default is 32.

        Returns:
            np.array: Transformed embeddings.

        Examples:
        >>> from sumire.vectorizer.transformer_emb import TransformerEmbeddingVectorizer
        >>> vectorizer = TransformerEmbeddingVectorizer()
        >>> texts = ["This is a sample sentence.", "Another example."]
        >>> vectors = vectorizer.transform(texts)
        >>> vectors.shape
        (2, 768)  # Assuming a BERT model with 768-dimensional embeddings
        """
        ret = []
        for batch in [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]:
            with torch.no_grad():
                self.model.eval()
                inputs = self.tokenizer(batch,
                                        return_tensors="pt",
                                        max_length=self.max_length,
                                        padding="max_length",
                                        truncation=True)
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(self.model.device)
                try:
                    outputs = self.model(**inputs, return_dict=True)
                except TypeError:  # line-distillbert got an unexpected keyword argument 'token_type_ids'.
                    outputs = self.model(inputs["input_ids"], return_dict=True)
                for k in inputs.keys():
                    inputs[k] = inputs[k].cpu().detach()
                if self.pooling_method == "cls":
                    last_hidden_state = outputs.last_hidden_state.cpu().detach()
                    cls_token_idx = inputs["input_ids"] == self.tokenizer.cls_token_id
                    if cls_token_idx.sum() == 0:
                        logger.info("No CLS token in the inputs, try using bos_token...")
                        cls_token_idx = inputs["input_ids"] == self.tokenizer.bos_token_id
                        assert cls_token_idx.sum() > 0, "No CLS, BoS token in the inputs."  # TODO
                    cls_emb = last_hidden_state[cls_token_idx].numpy()
                    ret.append(cls_emb)
                elif self.pooling_method in ["mean", "max"]:
                    for idx in range(outputs.last_hidden_state.shape[0]):
                        sample_hidden_state = outputs.last_hidden_state[idx][
                            inputs["input_ids"][idx] != self.tokenizer.pad_token_id]

                        if self.pooling_method == "mean":
                            pooled_emb = sample_hidden_state.mean(dim=0)
                        elif self.pooling_method == "max":
                            pooled_emb = sample_hidden_state.max(dim=0).values
                        ret.append(pooled_emb.numpy().reshape(1, -1))
        return np.concatenate(ret)

    def get_token_vectors(self, texts: Union[str, List[str]]) -> EncodeTokensOutputs:
        """
        Tokenizes each input text and obtains a tuple list of (token, token_vector) for each input text.

        Args:
            texts (TokenizerInputs): The input text or a list of texts to tokenize.

        Returns:
            EncodeTokensOutputs: Each internal list consists of a tuple of
                tokenized words and their vector representations.

        Examples:
            >>> import numpy as np
            >>> from sumire.vectorizer.transformer_emb import TransformerEmbeddingVectorizer
            >>> vectorizer = TransformerEmbeddingVectorizer()
            >>> texts = ["これはテスト文です。", "別のテキストもトークン化します。"]
            >>> vectors = vectorizer.get_token_vectors(texts)
            >>> len(vectors)
            2
            >>> isinstance(vectors[0][0][0], str)
            True
            >>> vectors[0][0][1].shape == (768, )
            True
        """
        ret = []
        with torch.no_grad():
            self.model.eval()
            for batch in [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]:
                inputs = self.tokenizer(batch,
                                        return_tensors="pt",
                                        max_length=self.max_length,
                                        padding="max_length",
                                        truncation=True)
                input_ids = inputs["input_ids"].to(self.model.device)
                tokenized = [self.tokenizer.convert_ids_to_tokens(i) for i in inputs["input_ids"].tolist()]
                outputs = self.model(input_ids, return_dict=True)
                input_ids = input_ids.cpu()
                last_hidden_states = outputs.last_hidden_state.cpu()
                for idx in range(last_hidden_states.shape[0]):
                    sample_hidden_state = last_hidden_states[idx][input_ids[idx] != self.tokenizer.pad_token_id]
                    tokens = [i for i in tokenized[idx] if i != self.tokenizer.pad_token]
                    part = []
                    for token, hidden_state in zip(tokens, sample_hidden_state):
                        part.append((token, hidden_state.numpy()))
                    ret.append(part)
        return ret

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        """
        Load a pretrained TransformerEmbeddingVectorizer from a specified path.

        Args:
            path (str or Path): The directory path to load the pretrained vectorizer from.

        Returns:
            TransformerEmbeddingVectorizer: The loaded pretrained vectorizer.

        """
        init_args = cls.load_init_args(path)
        obj = TransformerEmbeddingVectorizer(**init_args)
        return obj
