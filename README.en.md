# sumire

A Scikit-learn compatible Japanese word segmenter and text vectorization tool 
for CPU-based Japanese natural language processing that can be used 
without pre-installation of morphological analyzers or other tools.

[![PyPI - Version](https://img.shields.io/pypi/v/sumire.svg)](https://pypi.org/project/sumire)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sumire.svg)](https://pypi.org/project/sumire)
![Pytest](https://github.com/reiven-c-t/sumire/actions/workflows/python-app.yml/badge.svg)

-----

**Table of Contents**

- [sumire](#sumire)
  * [Installation](#installation)
    + [pre-requirements](#pre-requirements)
  * [Usage](#usage)
      - [Tokenizer usage](#tokenizer-usage)
      - [Vectorizer usage](#vectorizer-usage)
  * [Development background](#development-background)
      - [Unmotivated development tasks (at this moment.)](#unmotivated-development-tasks-at-this-moment)
      - [Roadmap (motivated development tasks)](#roadmap-motivated-development-tasks)
    + [Coding rule](#coding-rule)
  * [License](#license)
  * [Acknowledgements (Dependent libraries, data, and models.)](#acknowledgements-dependent-libraries-data-and-models)

## Installation

### pre-requirements

- Tested OS: ubuntu 22.04.
- python >=3.9
- make
- cmake
- git 

If you use MeCab-IPAdic-Neologd or MeCab-Unidic-Neologd, you may need to login to your git account (TODO: test).

```shell
# Jumanpp dependencies.
sudo apt update -y;
sudo apt install -y cmake libeigen3-dev libprotobuf-dev protobuf-c-compiler;
```

Just `pip install sumire`, and you can use MeCab and Jumanpp without installation.
If you do not have MeCab or Jumanpp binaries or dictionaries, 
this library will be installed these software in `$HOME/.local/sumire/` when you instantiate Tokenizer.


## Usage

#### Tokenizer usage

```python
from sumire.tokenizer import MecabTokenizer, JumanppTokenizer


text = "これはテスト文です。" 
texts = ["これはテスト文です。", "別のテキストもトークン化します。"]

mecab = MecabTokenizer("unidic-lite")
text_mecab_tokenized = mecab.tokenize(text)
texts_mecab_tokenized = mecab.tokenize(texts)

jumanpp = JumanppTokenizer()
jumanpp.tokenize(text)
text_jumanpp_tokenized = jumanpp.tokenize(text)
texts_jumanpp_tokenized = jumanpp.tokenize(texts)
```

#### Vectorizer usage

```python
from sumire.tokenizer.mecab import MecabTokenizer
from sumire.vectorizer.count import CountVectorizer
from sumire.vectorizer.swem import W2VSWEMVectorizer
from sumire.vectorizer.transformer_emb import TransformerEmbeddingVectorizer

texts = ["これはテスト文です。", "別のテキストもトークン化します。"]

count_vectorizer = CountVectorizer(MecabTokenizer())
swem_vectorizer = W2VSWEMVectorizer()
bert_cls_vectorizer = TransformerEmbeddingVectorizer()

# fit and transform at the same time. (Of course, you can .fit() and .transform() separately!)
count_vectorized = count_vectorizer.fit_transform(texts)
swem_vectorized = swem_vectorizer.fit_transform(texts)
bert_cls_vectorized = bert_cls_vectorizer.fit_transform(texts)

# save and load vectorizer.
count_vectorizer.save_pretrained("path/to/count_vectorizer")
count_vectorizer = CountVectorizer.from_pretrained("path/to/count_vectorizer")
swem_vectorizer.save_pretrained("path/to/swem_vectorizer")
swem_vectorizer = W2VSWEMVectorizer.from_pretrained("path/to/swem_vectorizer")
bert_cls_vectorizer.save_pretrained("path/to/bert_cls_vectorizer")
bert_cls_vectorizer = TransformerEmbeddingVectorizer.from_pretrained("path/to/beert_cls_vectorizer")
```

For detailed documentation of each tokenizer and vectorizer module, please refer to the [documentation page](https://reiven-c-t.github.io/sumire/).
See `/sumire/resources/model_card` for information on working models of Transformers and gensim.

## Development background

With the rise of LLM, more and more attention has been paid to 
practical tasks in Japanese NLP such as search, sentiment analysis, and other text classification and regression.
In these basic tasks, word segmentation of Japanese texts and 
obtaining distributed representations of words and sentences are among the most fundamental processes.
In the era of LLM, tuning of pre-trained Transformer models or
Embeddings by Open AI API are obviously the most important technologies in text distributed representations,
and can be easily implemented. 

However, in practical use, using state-of-the-art methods such as BERT and OpenAI API,
which require expensive GPUs or APIs that incur a cost per query,
is burdensome in terms of both computational and operational costs.
In addition, during the Proof of Concept (PoC) phase in the early stages of a project,
it is a little bit time-consuming to go through the ~~tedious~~ installation of dictionary data and morphological analyzers,
as well as the methods and properties of the slightly different APIs.

Based on these points, we have developed a library that allows users to 
quickly focus on modeling and analysis in PoC in an environment where GPUs are not always available,
by providing text to a unified API interface for each function like Scikit-learn.

#### Unmotivated development tasks (at this moment.)

- Using Open-AI Embedding model. (Because of its high price.) 
- For Embedding with pre-trained Transformer models, implement tuning functions that require a GPU. (Because I don't have a GPU at hand.)
- Reduce the internal readability of the library significantly for the sake of execution speed. 
  - In a small scale PoC, I think that the implementation speed is more important than the execution speed.
  - By maintaining the readability of the library so that each developer can easily remove unnecessary processes and customize the library, we believe that the library will help developers understand the code when speed or disk space becomes a problem in large-scale operation after the PoC.

#### Roadmap (motivated development tasks)

- vectorizer inputs decode() functions.
- Check on Google colab.

### Coding rule

<https://pep8-ja.readthedocs.io/ja/latest/>


## License

`sumire` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


## Acknowledgements (Dependent libraries, data, and models.)

See `dependent_licenses.csv`.