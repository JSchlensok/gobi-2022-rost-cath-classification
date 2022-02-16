from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal
import random

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.gobi_cath_classification.pipeline.utils.CATHLabel import CATHLabel

splits = ["train", "val", "test"]


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: List[str]
    train_labels: List[CATHLabel]
    X_val: np.ndarray
    y_val: List[str]
    X_test: np.ndarray
    y_test: List[str]

    def __post_init__(self):
        # Order labels
        self.train_labels = sorted(list(set(self.train_labels)))

        # Make datasets splits more accessible
        self._X = [self.X_train, self.X_val, self.X_test]
        self._y = [self.y_train, self.y_val, self.y_test]
        self._data = {
            'X': {split: data for split, data in zip(splits, self._X)},
            'y': {split: data for split, data in zip(splits, self._y)}
        }

    def shape(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        return {x_or_y: {split: self._data[x_or_y][split].shape for split in splits} for x_or_y in "Xy"}

    def getSplit(self, split: Literal["train", "val", "test"]) -> Tuple[np.ndarray, List[str]]:
        return self._data['X'][split], self._data['y'][split]

    ###################
    # BUILDER METHODS #
    ###################

    def shuffle(self, rng: np.random.RandomState) -> None:
        for X, y in zip(self._X, self._y):
            shuffled_indices = rng.permutation(len(X))
            X = X[shuffled_indices]
            y = [y[a] for a in shuffled_indices]

    def filter(self, cath_level: Literal['C', 'A', 'T', 'H']) -> None:
        valid_labels = [label[cath_level] for label in self.train_labels]
        for X, y in zip(self._X, self._y):
            filtered_data = [[embedding, label] for embedding, label in zip(X, y) if label[cath_level] in valid_labels]
            X, y = zip(*filtered_data)

    def scale(self) -> None:
        scaler = StandardScaler()
        scaler.fit(X=self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)
