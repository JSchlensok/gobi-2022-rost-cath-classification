from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from typing_extensions import Literal

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch

from gobi_cath_classification.pipeline.utils import CATHLabel

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

    def shape(self) -> Dict[str, Dict[str, Union[int, Tuple[int, int]]]]:
        return {
            "X": {"train": self.X_train.shape, "val": self.X_val.shape, "test": self.X_test.shape},
            "y": {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.X_test)},
        }

    def get_copy(self) -> Dataset:
        return deepcopy(self)

    def get_split(
        self,
        split: Literal["train", "val", "test"],
        as_tensors: bool = False,
        zipped: bool = False,
    ) -> Union[List[Tuple[np.ndarray, List[CATHLabel]], Tuple[np.ndarray, List[CATHLabel]]]]:
        data = {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
        }[split]

        if as_tensors:
            # convert X embedding to tensor
            x_tensor = torch.from_numpy(np.array(data[0]))
            data = (x_tensor, data[1])

        if zipped:
            return list(zip(data[0], data[1]))
        else:
            return data

    def get_filtered_version(self, cath_level: Literal["C", "A", "T", "H"]) -> Dataset:
        """
        Filter out all sequences from the validation & test set where there is no sequence sharing its CATH label
        up to the specified level
        """
        valid_labels = [label[cath_level] for label in self.train_labels]

        X_val, y_val = [
            list(unzipped)
            for unzipped in zip(
                *[
                    [embedding, label]
                    for embedding, label in zip(self.X_val, self.y_val)
                    if label[cath_level] in valid_labels
                ]
            )
        ]
        X_val = np.array(X_val)

        X_test, y_test = [
            list(unzipped)
            for unzipped in zip(
                *[
                    [embedding, label]
                    for embedding, label in zip(self.X_test, self.y_test)
                    if label[cath_level] in valid_labels
                ]
            )
        ]
        X_test = np.array(X_test)

        return Dataset(self.X_train, self.y_train, self.train_labels, X_val, y_val, X_test, y_test)

    ###################
    # BUILDER METHODS #
    ###################

    def shuffle(self, rng: np.random.RandomState) -> None:
        X_train, y_train = shuffle(self.X_train, self.y_train, random_state=rng)
        X_val, y_val = shuffle(self.X_val, self.y_val, random_state=rng)
        X_test, y_test = shuffle(self.X_test, self.y_test, random_state=rng)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def scale(self) -> None:
        scaler = StandardScaler()
        scaler.fit(X=self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)
