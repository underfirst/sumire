# sumire

**形態素解析器などの事前インストールなし**で使える, 
CPUベースの日本語自然言語処理のための,
Scikit-learn互換の日本語の単語分割器と, テキストのベクトル化ツール.

[![PyPI - Version](https://img.shields.io/pypi/v/sumire.svg)](https://pypi.org/project/sumire)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sumire.svg)](https://pypi.org/project/sumire)
![Lint](https://github.com/reiven-c-t/sumire/actions/workflows/python-lint.yml/badge.svg)
![Test](https://github.com/reiven-c-t/sumire/actions/workflows/python-test.yml/badge.svg)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/reiven-c-t/4300c0ff006aab09e6733925efbbf517/raw/3316d071e93ee9de175dc83086870cbd082c0b65/gistfile1.txt)(https://github.com/reiven-c-t/sumire/actions/workflows/python-test.yml)

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

MeCab-IPAdic-Neologdや, MeCab-Unidic-Neologdを使う場合, もしかしたらgitアカウントにログインが必要かもしれません (TODO: テスト).

```shell
# Jumanpp dependencies.
sudo apt update -y;
sudo apt install -y cmake libeigen3-dev libprotobuf-dev protobuf-c-compiler;
```

`pip install sumire`だけで, **MeCabもJumanppも, インストールなしで**使えます.
MeCabやJumanppの実行バイナリや各種辞書がなければ, `$HOME/.local/sumire/`にTokenizerをインスタンス化した時にインストールされます.

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

count_vectorizer = CountVectorizer()  # this automatically use MecabTokenizer()
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

各単語分割器や文分散表現モジュールの詳細なドキュメントは[ドキュメントページ](https://reiven-c-t.github.io/sumire/)を参照してください.
また, Transformersやgensimの動作済みmodelの情報は, `/sumire/resources/model_card`を参照してください.


## Development background

LLMの隆盛に伴い, 検索, 感情分析, その他テキスト分類・回帰などの日本語のNLPの実用タスクへの注目も高まりつつある.
これらの基本的なタスクにおいて, 日本語のテキストを単語分割や, 単語や文の分散表現を得ることは, 最も基礎的な処理の一つである.
LLMの時代において, 事前訓練済みTransformerモデルのチューニングや, Open AI APIによるEmbeddingsは,
テキスト分散表現技術において最も重要な技術であることはいうまでもなく, また, 簡単に実装できるといえば実装できる. 

しかし, 実用の現場において, BERTや, OpenAI APIなどの, 
高価なGPUが必要な手法や, 1 Queryごとに費用が発生するAPIを用いた最先端の手法を使うことは, 計算量・運用コストの両面から負荷が高い.
また, データセット構築段階などのプロジェクトの初期段階での概念実証 (PoC) において, 
辞書データや形態素解析器の~~めんどくさい~~インストール作業や,
それぞれやや異なるAPIのメソッドやプロパティを調べながら作業を行うのは少しばかり手間である.

これらの点を踏まえて, GPUがあるとは限らない手元環境で, 
PoCにおけるモデリング・分析部分へ速やかに注力できように,
Scikit-learnのように, 機能ごとに統一的なAPIインターフェースで, 
テキストを与えればとりあえず色々な文の分散表現を取得できるライブラリを開発した.

#### Unmotivated development tasks (at this moment.)

- Open-AI Embedding modelを使うこと. (高い.)
- 事前訓練済みTransformerモデルによるEmbeddingについて, GPUが必要なチューニング機能を実装すること. (手元にGPUがない.)
- 実行速度のためにライブラリ内部の可読性を大きく下げること. 
  - 小規模なPoCにおいて, コードの実行速度より, 実装速度のほうが重要である.
  - PoC後の大規模な運用にて, 速度やディスク容量が問題になった場合があれば, 
    本ライブラリ中の不要な処理をそれぞれの開発者が削除したりカスタマイズしやすいように, 可読性を維持することが望ましい.


#### Roadmap (motivated development tasks)

- vectorizer inputsのdecode().
- Google colabでの動作環境検証.

### Coding rule

<https://pep8-ja.readthedocs.io/ja/latest/>


## License

`sumire` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


## Acknowledgements (Dependent libraries, data, and models.)

See `dependent_licenses.csv`.