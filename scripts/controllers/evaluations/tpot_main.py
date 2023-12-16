import json
from os import makedirs
from pathlib import Path
from random import seed

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from tpot import TPOTClassifier

from scripts.data.jglue_data_loader import JGLUEDataLoader
from sumire.tokenizer import MecabTokenizer
from sumire.vectorizer import CountVectorizer, TfidfVectorizer, TransformerEmbeddingVectorizer, W2VSWEMVectorizer

classifier_config_dict = {
    # Classifiers
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.1, 0.91, 0.1),
        'min_samples_split': range(2, 21, 5),
        'min_samples_leaf': range(1, 21, 5),
        'bootstrap': [True]
    },

    'sklearn.neural_network.MLPClassifier': {
        'hidden_layer_sizes': [(100, ), (100, 100,)],
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'activation': ['relu', 'logistic', 'tanh'],
        'early_stopping': [True],
        'learning_rate_init': [1e-3, 1e-2, 1e-1],
        'learning_rate': ["adaptive"]
    },

    # Preprocesssors
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.1)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    # Selectors
    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.1, 0.51, 0.1),
        'n_features_to_select': np.arange(0.25, 0.751, 0.25),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },
}


def valid_json_exists(json_path: Path) -> bool:
    if json_path.exists():
        with open(json_path) as f:
            try:
                json.load(f)
                return True
            except json.JSONDecodeError:
                pass
    return False


def evaluate_report(model, trainX, train_y, validX, valid_y, json_path):
    valid_pred_y = model.predict(validX)
    train_pred_y = model.predict(trainX)

    train_acc = accuracy_score(train_y, train_pred_y)
    train_f1 = f1_score(train_y, train_pred_y, average="macro")
    val_acc = accuracy_score(valid_y, valid_pred_y)
    val_f1 = f1_score(valid_y, valid_pred_y, average="macro")

    makedirs(json_path.parent, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({
            "num_train_samples": trainX.shape[0],
            "num_valid_samples": validX.shape[0],
            "train_acc": train_acc,
            "train_f1": train_f1,
            "valid_acc": val_acc,
            "valid_f1": val_f1},
            f, ensure_ascii=False, indent=2)


def experiment(loader, init_vectorizer, json_path: Path, SEED=0):
    seed(SEED)
    np.random.seed(SEED)
    makedirs(json_path.parent, exist_ok=True)

    logger.info(f"Start {json_path.parent.name}.")
    logger.info(f"Start vectorizer {vectorizer_setting_name}.")
    vectorizer = init_vectorizer()

    logger.info("Start vectorize data.")
    (trainX, train_y), (validX, valid_y) = loader.convert_dataset_by_vectorizer(vectorizer)
    vectorizer.save_pretrained(json_path.parent)
    del vectorizer

    logger.info("Start tuning.")
    model = TPOTClassifier(random_state=SEED,
                           generations=5,
                           population_size=10,
                           # max_time_mins=60,
                           max_eval_time_mins=30,
                           config_dict=classifier_config_dict,
                           verbosity=3)
    model.fit(trainX, train_y)

    logger.info("Start evaluation.")
    evaluate_report(model, trainX, train_y, validX, valid_y, json_path)
    model.export(json_path.parent / 'tpot_exported_pipeline.py')


if __name__ == '__main__':
    EVALUATION_RESULT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "evaluation"
    LOG_DIR = EVALUATION_RESULT_DIR / "logs"
    makedirs(LOG_DIR, exist_ok=True)
    logger.add(LOG_DIR / "tpot_main{time}.log")

    vectorizer_settings = [
        (lambda: TransformerEmbeddingVectorizer(), "cl-tohoku-bert-japanese-v3"),
        (lambda: TransformerEmbeddingVectorizer(pooling_method="mean"), "cl-tohoku-bert-japanese-v3-mean"),
        (lambda: TransformerEmbeddingVectorizer(pooling_method="mean"), "sonoisa/sentence-bert-base-ja-mean-tokens-v2"),
        (lambda: W2VSWEMVectorizer("cc.ja.300"), "cc.ja.300"),
        (lambda: W2VSWEMVectorizer("cc.ja.300", pooling_method="max"), "cc.ja.300-max"),
        (lambda: W2VSWEMVectorizer("jawiki.all_vectors.300d"), "jawiki.all_vectors.300d"),
        (lambda: W2VSWEMVectorizer("jawiki.all_vectors.300d", pooling_method="max"), "jawiki.all_vectors.300d"),
        (lambda: CountVectorizer(MecabTokenizer()), "count_mecab"),
        (lambda: TfidfVectorizer(MecabTokenizer()), "tfidf_mecab"),
    ]

    for task in [ "MARC-ja", "JCoLA", "JNLI",]:
        loader = JGLUEDataLoader(task)
        loader.set_data()
        logger.info(f"Start {task}")

        for init_vectorizer, vectorizer_setting_name in vectorizer_settings:
            json_path = EVALUATION_RESULT_DIR / f"{task}/{vectorizer_setting_name.replace('/', '_')}/result.json"
            if valid_json_exists(json_path):
                logger.info(f"Experiment result already exists in {json_path}")
                continue

            experiment(loader, init_vectorizer, json_path)
