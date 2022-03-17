from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Any
from typing_extensions import Literal

from nptyping import NDArray, Float
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
from torchtyping import TensorType

from gobi_cath_classification.pipeline.utils import CATHLabel

splits = ["train", "val", "test"]

@dataclass
class Dataset:
    X_train: NDArray[(Any, 1024), Float]
    y_train: List[CATHLabel]
    train_labels: List[CATHLabel]
    X_val: NDArray[(Any, 1024), Float]
    y_val: List[CATHLabel]
    X_test: NDArray[(Any, 1024), Float]
    y_test: List[CATHLabel]
    stores_strings: bool = False
    X_train_str: List[str] = None
    X_val_str: List[str] = None
    X_test_str: List[str] = None

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
        x_encoding: Literal["embedding-array", "embedding-tensor", "string"] = "embedding-array",
        zipped: bool = False,
    ) -> Union[
        List[Tuple[NDArray[(Any, 1024), Float], CATHLabel]],
        Tuple[NDArray[(Any, 1024), Float], List[CATHLabel]],
        List[Tuple[TensorType[Any, 1024], CATHLabel]],
        Tuple[TensorType[Any, 1024], List[CATHLabel]],
        List[Tuple[str, CATHLabel]],
        Tuple[List[str], List[CATHLabel]],
    ]:
        x, y = {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
        }[split]

        if x_encoding == "embedding-tensor":
            # convert X embeddings to tensors
            x = torch.from_numpy(np.array(x))

        elif x_encoding == "string":
            if not self.stores_strings:
                raise ValueError(
                    "String representation requested, but no strings loaded. Use Dataset.load_strings() before calling get_split()"
                )
            x = {"train": self.X_train_str, "val": self.X_val_str, "test": self.X_test_str}[split]

        if zipped:
            return list(zip(x, y))
        else:
            return x, y

    def get_filtered_version(self, cath_level: Literal["C", "A", "T", "H"]) -> Dataset:
        """
        Filter out all sequences from the validation & test set where there is no sequence sharing its CATH label
        up to the specified level
        """
        valid_labels = [label[cath_level] for label in self.train_labels]

        def _filter_x_based_on_y(X, y):
            x_filtered, y_filtered = [
                list(unzipped)
                for unzipped in zip(
                    *[
                        [embedding, label]
                        for embedding, label in zip(X, y)
                        if label[cath_level] in valid_labels
                    ]
                )
            ]
            return x_filtered, y_filtered

        X_val, y_val = _filter_x_based_on_y(self.X_val, self.y_val)
        X_val = np.array(X_val)

        X_test, y_test = _filter_x_based_on_y(self.X_test, self.y_test)
        X_test = np.array(X_test)

        copy = Dataset(self.X_train, self.y_train, self.train_labels, X_val, y_val, X_test, y_test)

        if self.stores_strings:
            X_val_str, _ = _filter_x_based_on_y(self.X_val_str, self.y_val)
            X_test_str, _ = _filter_x_based_on_y(self.X_test_str, self.y_test)
            copy.load_strings(self.strings["train"], X_val_str, X_test_str)

        return copy

    ###################
    # BUILDER METHODS #
    ###################

    def shuffle_training_set(self, rng: np.random.RandomState) -> None:
        X_train, y_train = shuffle(self.X_train, self.y_train, random_state=rng)

        self.X_train = X_train
        self.y_train = y_train

    def scale(self) -> None:
        scaler = StandardScaler()
        scaler.fit(X=self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

    def load_strings(self, train: List[str], val: List[str], test: List[str]) -> None:
        self.stores_strings = True
        self.X_train_str = train
        self.X_val_str = val
        self.X_test_str = test
