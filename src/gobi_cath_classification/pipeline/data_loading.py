from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

from gobi_cath_classification.pipeline.utils import CATHLabel
from gobi_cath_classification.pipeline.data import Dataset
import gobi_cath_classification.pipeline.data.data_loading as new

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"


@dataclass
class DataSplits:
    """
    Interface to new data.Dataset class for compatibility
    """

    X_train: np.ndarray
    y_train: List[str]
    X_val: np.ndarray
    y_val: List[str]
    X_test: np.ndarray
    y_test: List[str]
    all_labels_train_sorted: List[str]

    def __post_init__(self):
        self._dataset = Dataset(
            self.X_train,
            self.y_train,
            [CATHLabel(label) for label in self.all_labels_train_sorted],
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )

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
        assert cath_level in ["C", "A", "T", "H"]
        filtered_data = deepcopy(self._dataset)
        filtered_data.filter(cath_level)
        return filtered_data.getSplit("val")

    def get_filtered_test_set_for_level(self, cath_level: str) -> Tuple[np.ndarray, List[str]]:
        assert cath_level in ["C", "A", "T", "H"]
        filtered_data = deepcopy(self._dataset)
        filtered_data.filter(cath_level)
        return filtered_data.getSplit("test")

    def _get_filtered_set_for_level(
        self, X: np.ndarray, y: List[str], cath_level: str = "T"
    ) -> Tuple[np.ndarray, np.ndarray]:

        if cath_level not in ["C", "A", "T", "H"]:
            raise ValueError(f"Invalid CATH level: {cath_level}")

        valid_labels = [label[cath_level] for label in self._dataset.train_labels]
        filtered_data = [
            [embedding, label]
            for embedding, label in zip(X, y)
            if CATHLabel(label)[cath_level] in valid_labels
        ]
        X, y = zip(*filtered_data)

        return X, y

    def shuffled(
        self, rng: np.random.RandomState, shuffle_train=True, shuffle_val=True, shuffle_test=True
    ):
        """

        Returns a new DataSplits object with shuffled trainings set and/or shuffled validation set
        and/or shuffled test set.

        """
        shuffled_data = deepcopy(self._dataset)
        shuffled_data.shuffle(rng)
        if not shuffle_train:
            shuffled_data.X_train = self._dataset.X_train
            shuffled_data.y_train = self._dataset.y_train

        if not shuffle_val:
            shuffled_data.X_val = self._dataset.X_val
            shuffled_data.y_val = self._dataset.y_val

        if not shuffle_test:
            shuffled_data.X_test = self._dataset.X_test
            shuffled_data.y_test = self._dataset.y_test

        return DataSplits(
            shuffled_data.X_train,
            [str(label) for label in shuffled_data.y_train],
            shuffled_data.X_val,
            [str(label) for label in shuffled_data.y_val],
            shuffled_data.X_test,
            [str(label) for label in shuffled_data.y_test],
            [str(label) for label in shuffled_data.train_labels],
        )


def label_for_level(label: str, cath_level: str) -> str:
    """
    Example 1:
        Input: label_for_level(label="3.250.40.265", cath_level="H")
        Output: "3.250.40.265"

    Example 2:
        Input: label_for_level(label="3.250.40.265", cath_level="A")
        Output: "3.250"
    """
    return str(CATHLabel(label)[cath_level])


def read_in_sequences(path_to_file: Path) -> Dict[str, str]:
    return read_in_sequences(path_to_file)


def read_in_embeddings(path_to_file: Path) -> Dict[str, np.ndarray]:
    return read_in_embeddings(path_to_file)


def read_in_labels(path_to_file: Path) -> Dict[str, str]:
    df = pd.read_csv(filepath_or_buffer=path_to_file, delimiter=r"\s+", header=None, comment="#")

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
    rng: np.random.RandomState,
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

    id2seqs_all = {**id2seqs_train, **id2seqs_val, **id2seqs_test}
    print(f"len(id2seqs_train) = {len(id2seqs_train)}")
    print(f"len(id2seqs_val) = {len(id2seqs_val)}")
    print(f"len(id2seqs_test) = {len(id2seqs_test)}")
    print(f"len(id2seqs_all) = {len(id2seqs_all)}")

    print("Reading in Labels ...")
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
    print("Removing duplicates and mismatched entries ...")
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

    dataset = DataSplits(
        X_train=np.array([embeddings[cath_id] for cath_id in id2seqs_train.keys()]),
        y_train=[id2label[cath_id] for cath_id in id2seqs_train.keys()],
        X_val=np.array([embeddings[cath_id] for cath_id in id2seqs_val.keys()]),
        y_val=[id2label[cath_id] for cath_id in id2seqs_val.keys()],
        X_test=np.array([embeddings[cath_id] for cath_id in id2seqs_test.keys()]),
        y_test=[id2label[cath_id] for cath_id in id2seqs_test.keys()],
        all_labels_train_sorted=sorted(list(set([id2label[k] for k in id2seqs_train.keys()]))),
    )

    if shuffle_data:
        return dataset.shuffled(rng)
    else:
        return dataset


def scale_dataset(dataset: DataSplits):
    print("Scaling data ...")
    dataset._dataset.scale()
    return dataset
