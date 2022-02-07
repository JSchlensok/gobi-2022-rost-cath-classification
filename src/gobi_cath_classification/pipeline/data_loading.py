import random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: List[str]
    X_val: np.ndarray
    y_val: List[str]
    X_test: np.ndarray
    y_test: List[str]
    all_labels_train_sorted: List[str]

    def get_shape(self):
        return (
            self.X_train.shape,
            np.array(self.y_train).shape,
            self.X_val.shape,
            np.array(self.y_val).shape,
            self.X_test.shape,
            np.array(self.y_test).shape,
        )

    def get_filtered_validation_set_for_level(
        self, cath_level: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Args:
            cath_level:
                0 for 'C' in 'CATH'
                1 for 'A' in 'CATH'
                2 for 'T' in 'CATH'
                3 for 'H' in 'CATH'

        Returns:
            Returns a filtered validation set where for each instance there is at least one
            instance with that label in the training set. In other words, we remove all
            sequences that have a label that does not occur in the training set.
        """
        validation_set = self._get_filtered_set_for_level(
            X=self.X_val, y=self.y_val, cath_level=cath_level
        )
        return validation_set

    def get_filtered_test_set_for_level(
        self, cath_level: str
    ) -> Tuple[np.ndarray, List[str]]:

        test_set = self._get_filtered_set_for_level(
            X=self.X_val, y=self.y_val, cath_level=cath_level
        )
        return test_set

    def _get_filtered_set_for_level(
        self, X: np.ndarray, y: List[str], cath_level: str = "T"
    ) -> Tuple[np.ndarray, np.ndarray]:

        if cath_level not in ["C", "A", "T", "H"]:
            raise ValueError(f"Invalid CATH level: {cath_level}")

        X_filtered = []
        y_filtered = []

        training_labels_set = [
            label_for_level(label=label, cath_level=cath_level)
            for label in self.all_labels_train_sorted
        ]

        for index in range(len(y)):
            label_prefix = label_for_level(label=y[index], cath_level=cath_level)
            if label_prefix in training_labels_set:
                X_filtered.append(X[index])
                y_filtered.append(y[index])

        return np.array(X_filtered), y_filtered

    def get_index_for_label(self, label: str) -> int:
        index = self.all_labels_train_sorted.index(label)
        return index

    def get_label_for_index(self, index: int) -> str:
        label = self.all_labels_train_sorted[index]
        return label

    def shuffled(
        self, shuffle_train=True, shuffle_val=True, shuffle_test=True, random_seed=1
    ):
        """

        Returns a new DataSplits object with shuffled trainings set and/or shuffled validation set
        and/or shuffled test set.

        """
        new_datasplit = DataSplits(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
            all_labels_train_sorted=self.all_labels_train_sorted,
        )

        if shuffle_train:
            train_zipped = list(zip(self.X_train, self.y_train))
            random.Random(random_seed).shuffle(train_zipped)
            x, y = zip(*train_zipped)
            new_datasplit.X_train = np.array(x)
            new_datasplit.y_train = y

        if shuffle_val:
            val_zipped = list(zip(self.X_val, self.y_val))
            random.Random(random_seed).shuffle(val_zipped)
            x, y = zip(*val_zipped)
            new_datasplit.X_val = np.array(x)
            new_datasplit.y_val = y

        if shuffle_test:
            test_zipped = list(zip(self.X_test, self.y_test))
            random.Random(random_seed).shuffle(test_zipped)
            x, y = zip(*test_zipped)
            new_datasplit.X_test = np.array(x)
            new_datasplit.y_test = y

        assert self.get_shape() == new_datasplit.get_shape()

        return new_datasplit


def label_for_level(label: str, cath_level: str) -> str:
    """
    Example 1:
        Input: label_for_level(label="3.250.40.265", cath_level=3)
        Output: "3.250.40.265"

    Example 2:
        Input: label_for_level(label="3.250.40.265", cath_level=1)
        Output: "3.250."
    """
    check_if_cath_level_is_valid(cath_level=cath_level)

    level = "CATH".index(cath_level)
    label_for_level = ".".join(label.split(".")[: level + 1])

    return label_for_level


def check_if_cath_level_is_valid(cath_level: str):
    if cath_level not in ["C", "A", "T", "H"]:
        raise ValueError(f"Invalid CATH level: {cath_level}")


def read_in_sequences(path_to_file: Path) -> Dict[str, str]:
    id2seq = {}
    tmp_id = ""

    for line in open(path_to_file, "r"):
        if line.startswith(">"):
            tmp_id = line.replace(">", "").rstrip()
            # tmp_id = line.split("|")[-1].split("/")[0]

        elif not line.startswith("#"):
            tmp_seq = line.rstrip()
            id2seq[tmp_id] = tmp_seq

    return id2seq


def read_in_embeddings(path_to_file: Path) -> Dict[str, np.ndarray]:
    id2embedding = {}
    h5_file = h5py.File(path_to_file)

    for key, value in h5_file.items():
        protein_id = key.split("|")[-1].split("_")[0]
        values = value[()]
        id2embedding[protein_id] = values

    return id2embedding


def read_in_labels(path_to_file: Path) -> Dict[str, str]:
    df = pd.read_csv(
        filepath_or_buffer=path_to_file, delimiter=r"\s+", header=None, comment="#"
    )

    id2label = {}

    for index, row in df.iterrows():
        cath_id = row[0]

        c = str(row[1])  # class
        a = str(row[2])  # architecture
        t = str(row[3])  # topology
        h = str(row[4])  # homology

        label = ".".join([c, a, t, h])
        # label = np.array([c, a, t, h])
        id2label[cath_id] = label

    return id2label


def load_data(
    data_dir: Path,
    without_duplicates: bool,
    shuffle_data: bool = True,
    load_only_small_sample=False,
):
    print(f"Loading data from directory: {data_dir}")

    path_sequences_train = data_dir / "train74k.fasta"
    path_sequences_val = data_dir / "val200.fasta"
    path_sequences_test = data_dir / "test219.fasta"

    path_embeddings = data_dir / "cath_v430_dom_seqs_S100_161121.h5"
    path_labels = data_dir / "cath-domain-list.txt"

    if load_only_small_sample:
        path_sequences_train = data_dir / "sample_data/sample_train100.fasta"
        path_labels = data_dir / "sample_data/sample_cath-domain-list100.txt"

    print("Reading in Sequences ...")
    id2seqs_train = read_in_sequences(path_sequences_train)
    id2seqs_val = read_in_sequences(path_sequences_val)
    id2seqs_test = read_in_sequences(path_sequences_test)

    id2seqs_all = merge_two_dicts(
        id2seqs_train, merge_two_dicts(id2seqs_val, id2seqs_test)
    )
    print(f"len(id2seqs_train) = {len(id2seqs_train)}")
    print(f"len(id2seqs_val) = {len(id2seqs_val)}")
    print(f"len(id2seqs_test) = {len(id2seqs_test)}")
    print(f"len(id2seqs_all) = {len(id2seqs_all)}")

    print("Reading in Labels  ...")
    id2label_all = read_in_labels(path_to_file=path_labels)
    id2label = {}
    for key in id2seqs_all.keys():
        id2label[key] = id2label_all[key]

    print(f"len(id2label_all) = {len(id2label_all)}")
    print(f"len(id2label) = {len(id2label)}")

    print("Reading in Embeddings ...")
    id2embedding = read_in_embeddings(path_to_file=path_embeddings)
    embeddings = {}
    for key in id2seqs_all.keys():
        embeddings[key] = id2embedding[key]

    # remove duplicates and mismatched entries
    seq2count = Counter(id2seqs_all.values())
    for key_seq, value_count in seq2count.items():
        if value_count > 1:
            cath_ids = [x for x, y in id2seqs_all.items() if y == key_seq]

            # remove mismatched entries
            cath_labels = [id2label[cath_id] for cath_id in cath_ids]
            if len(set(cath_labels)) > 1:
                for cath_id in cath_ids:
                    id2seqs_train.pop(cath_id, None)
                    id2seqs_val.pop(cath_id, None)
                    id2seqs_test.pop(cath_id, None)
                    id2seqs_all.pop(cath_id, None)
                    id2label.pop(cath_id, None)
                    id2embedding.pop(cath_id, None)

            # remove duplicates
            elif without_duplicates:
                for cath_id in cath_ids[1:]:
                    id2seqs_train.pop(cath_id, None)
                    id2seqs_val.pop(cath_id, None)
                    id2seqs_test.pop(cath_id, None)
                    id2seqs_all.pop(cath_id, None)
                    id2label.pop(cath_id, None)
                    id2embedding.pop(cath_id, None)

    # +
    dataset = DataSplits(
        X_train=np.array([embeddings[cath_id] for cath_id in id2seqs_train.keys()]),
        y_train=[id2label[cath_id] for cath_id in id2seqs_train.keys()],
        X_val=np.array([embeddings[cath_id] for cath_id in id2seqs_val.keys()]),
        y_val=[id2label[cath_id] for cath_id in id2seqs_val.keys()],
        X_test=np.array([embeddings[cath_id] for cath_id in id2seqs_test.keys()]),
        y_test=[id2label[cath_id] for cath_id in id2seqs_test.keys()],
        all_labels_train_sorted=sorted(
            list(set([id2label[k] for k in id2seqs_train.keys()]))
        ),
    )
    if shuffle_data:
        dataset = dataset.shuffled()

    return dataset


def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z
