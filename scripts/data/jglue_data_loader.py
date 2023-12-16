from typing import Tuple

from loguru import logger
import numpy as np
from datasets import load_dataset

from sumire.vectorizer.base.common import BaseVectorizer

DataPair = Tuple[np.array, np.array]  # X, y
DatasetOutput = Tuple[DataPair, DataPair]


class JGLUEDataLoader:
    def __init__(self, name: str):
        self.name = name
        self.dataset = load_dataset("shunk031/JGLUE", name=name)
        self.texts = {"train1": [], "train2": [], "validation1": [], "validation2": []}
        self.labels = {"train": [], "validation": []}
        self.label_keys = []

    def set_data(self):
        for key in ["train", "validation"]:
            for row in self.dataset[key]:
                if self.name in ["MARC-ja", "JCoLA"]:
                    self.texts[key + "1"].append(row["sentence"])
                    self.labels[key].append(row["label"])
                    self.label_keys.append(row["label"])
                elif self.name in ["JNLI", "JSTS"]:
                    self.texts[key + "1"].append(row["sentence1"])
                    self.texts[key + "2"].append(row["sentence2"])
                    self.labels[key].append(row["label"])
                    if self.name == "JNLI":
                        self.label_keys.append(row["label"])
                else:
                    raise ValueError(f"{self.name} is invalid task name.")

    def convert_dataset_by_vectorizer(self, vectorizer: BaseVectorizer, dropna:bool=False) -> DatasetOutput:
        vectorizer.fit(self.texts["train1"] + self.texts["train2"])
        trainX = vectorizer.transform(self.texts["train1"])
        validX = vectorizer.transform(self.texts["validation1"])
        feature_length = trainX.shape[-1]
        train_nan_index = (trainX == 0).sum(axis=1) == feature_length
        valid_nan_index = (validX == 0).sum(axis=1) == feature_length
        if len(self.texts["train2"]) > 0:
            trainX2 = vectorizer.transform(self.texts["train2"])
            validX2 = vectorizer.transform(self.texts["validation2"])
            train2_nan_index = (trainX2 == 0).sum(axis=1) == feature_length
            train_nan_index = (train_nan_index.astype(int) + train2_nan_index.astype(int)) >= 1
            valid2_nan_index = (validX2 == 0).sum(axis=1) == feature_length
            valid_nan_index = (valid_nan_index.astype(int) + valid2_nan_index.astype(int)) >= 1
            trainX = np.concatenate([trainX, trainX2], axis=1)
            validX = np.concatenate([validX, validX2], axis=1)

        if self.name != "JSTS":
            train_y = np.array(self.labels["train"], dtype=int)
            valid_y = np.array(self.labels["validation"], dtype=int)
        else:
            train_y = np.array(self.labels["train"], dtype=float)
            valid_y = np.array(self.labels["validation"], dtype=float)

        # remove nan vector.
        num_drop = train_nan_index.sum()
        if num_drop > 0 and dropna:
            logger.info(f"Drops {num_drop} samples from test data, because it cannot vectorize.")
            trainX = trainX[train_nan_index.astype(int) != 0]
            train_y = train_y[train_nan_index.astype(int) != 0]

        num_drop = valid_nan_index.sum()
        if num_drop > 0 and dropna:
            logger.info(f"Drops {num_drop} samples from valid data, because it cannot vectorize.")
            validX = validX[valid_nan_index.astype(int) != 0]
            valid_y = valid_y[valid_nan_index.astype(int) != 0]
        logger.info(f"shape of train: {trainX.shape}, valid: {validX.shape}")

        return (trainX, train_y), (validX, valid_y)
