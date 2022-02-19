from dataclasses import dataclass
from typing import List, Dict, Tuple
from typing_extensions import Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.gobi_cath_classification.pipeline.utils.CATHLabel import CATHLabel

splits = ["train", "val", "test"]


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: List[CATHLabel]
    train_labels: List[CATHLabel]
    X_val: np.ndarray
    y_val: List[CATHLabel]
    X_test: np.ndarray
    y_test: List[CATHLabel]

    def __post_init__(self):
        # Order labels
        self.train_labels = sorted(list(set(self.train_labels)))

    def shape(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        return {
            "X": {"train": self.X_train.shape, "val": self.X_val.shape, "test": self.X_test.shape},
            "y": {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.X_test)}
        }

    def getSplit(self, split: Literal["train", "val", "test"]) -> Tuple[np.ndarray, List[CATHLabel]]:
        return {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test)
        }[split]

    ###################
    # BUILDER METHODS #
    ###################

    def shuffle(self, rng: np.random.RandomState) -> None:
        for X, y in [[self.X_train, self.y_train], [self.X_val, self.y_val], [self.X_test, self.y_test]]:
            shuffled_indices = rng.permutation(len(X))
            X = X[shuffled_indices]
            y = [y[a] for a in shuffled_indices]

    def filter(self, cath_level: Literal["C", "A", "T", "H"]) -> None:
        """
        Filter out all sequences from the validation & test set where there is no sequence sharing its CATH label
        up to the specified level
        """
        valid_labels = [label[cath_level] for label in self.train_labels]

        self.X_val, self.y_val = [list(unzipped) for unzipped in zip(*[
            [embedding, label] for embedding, label in zip(self.X_val, self.y_val)
            if label[cath_level] in valid_labels
        ])]

        self.X_test, self.y_test = [list(unzipped) for unzipped in zip(*[
            [embedding, label] for embedding, label in zip(self.X_test, self.y_test)
            if label[cath_level] in valid_labels
        ])]

    def scale(self) -> None:
        scaler = StandardScaler()
        scaler.fit(X=self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)
