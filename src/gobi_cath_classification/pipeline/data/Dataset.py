from __future__ import annotations

import collections
from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from typing_extensions import Literal

from gobi_cath_classification.pipeline.data.data_loading import (
    read_in_embeddings,
    read_in_sequences,
    read_in_labels,
)
from gobi_cath_classification.pipeline.utils.torch_utils import RANDOM_SEED
from gobi_cath_classification.pipeline.utils import CATHLabel

splits = ["train", "val", "test"]

# TODO test & debug


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: List[CATHLabel]
    X_val: np.ndarray
    y_val: List[CATHLabel]
    X_test: np.ndarray
    y_test: List[CATHLabel]
    X_tmp_holdout: Optional[np.ndarray]
    y_tmp_holdout: Optional[List[CATHLabel]]
    X_lookup: Optional[np.ndarray]
    y_lookup: Optional[List[CATHLabel]]

    stores_strings: bool = False
    X_train_str: List[str] = None
    X_val_str: List[str] = None
    X_test_str: List[str] = None
    X_tmp_holdout_str: List[str] = None
    X_lookup_str: List[str] = None

    label_encoder: LabelEncoder = None

    def __post_init__(self):
        # Order labels
        self.train_labels = sorted(list(set(self.y_train)))
        self.all_labels = list({*self.y_train, *self.y_val, *self.y_test})

        if self.y_tmp_holdout:
            self.all_labels = list({*self.all_labels, *self.y_tmp_holdout})

        if self.y_lookup:
            self.all_labels = list({*self.all_labels, *self.y_lookup})

    @classmethod
    def load(
        cls,
        data_dir: Path,
        rng: np.random.RandomState,
        without_duplicates: bool,
        load_tmp_holdout_set: bool,
        load_lookup_set: bool = False,
        load_strings: bool = False,
        load_only_small_sample: bool = False,
        shuffle_data: bool = True,
        reloading_allowed: bool = True,
        # Load data with Y-values only being one level
        specific_level: Literal["C", "A", "T", "H"] = None,
        # Load data with Y-values only up to the given cutoff level
        level_cutoff: Literal["C", "A", "T", "H"] = None,
        encode_labels: bool = False,
    ):
        logging.info(f"Loading data from directory: {data_dir}")
        sequence_paths: Dict[str, Path] = {
            "train": data_dir / "train74k.fasta",
            "val": data_dir / "val200.fasta",
            "test": data_dir / "test219.fasta",
            "holdout": data_dir / "holdout389.fasta",
            "lookup": data_dir / "lookup78k.fasta",
        }
        embedding_paths: Dict[str, Path] = {"all": data_dir / "cath_v430_dom_seqs_S100_161121.h5"}
        label_path = data_dir / "cath-domain-list.txt"

        if load_tmp_holdout_set:
            embedding_paths["holdout"] = data_dir / "temporal_holdout_set.h5"
            label_path = data_dir / "cath-domain-list-updated.txt"

        if load_only_small_sample:
            sequence_paths["train"] = data_dir / "sample_data/sample_train100.fasta"
            label_path = data_dir / "sample_data/sample_cath-domain-list100.txt"

        # Reload if possible
        duplicates_tag = "no-duplicates" if without_duplicates else "duplicates"
        small_sample_tag = "small_sample" if load_only_small_sample else "full"
        tmp_holdout_tag = "with_tmp_holdout" if load_tmp_holdout_set else "no-tmp_holdout"
        lookup_set_tag = "_with_lookup" if load_lookup_set else ""
        including_strings_tag = "_with_strings" if load_strings else ""
        label_encoding_tag = "_with_label_encoding" if encode_labels else ""
        serialized_dataset_location = f"serialized_dataset_{duplicates_tag}_{small_sample_tag}{lookup_set_tag}{including_strings_tag}{label_encoding_tag}_{tmp_holdout_tag}.pickle "

        if reloading_allowed:
            logging.info("Trying to find a serialized dataset ...")

            if (data_dir / serialized_dataset_location).exists():
                logging.info(
                    f"Found a serialized dataset at {data_dir / serialized_dataset_location} to save time"
                )
                with open(data_dir / serialized_dataset_location, "rb") as f:
                    serialized_dataset = pickle.load(f)

                if shuffle_data:
                    serialized_dataset.shuffle_training_set(np.random.RandomState(RANDOM_SEED))

                return serialized_dataset

            else:
                logging.info(
                    f"Couldn't find a matching serialized dataset, creating a new one at {data_dir / serialized_dataset_location}"
                )

        logging.info("Reading in sequences ...")
        id2seqs = {
            split: read_in_sequences(sequence_paths[split]) for split in ["train", "val", "test"]
        }

        if load_tmp_holdout_set:
            id2seqs["holdout"] = read_in_sequences(sequence_paths["holdout"])

        if load_lookup_set:
            id2seqs["lookup"] = read_in_sequences(sequence_paths["lookup"])

        id2seqs["all"] = {
            id: seq for id2seqs_split in id2seqs.values() for id, seq in id2seqs_split.items()
        }

        for split, id2seqs_split in id2seqs.items():
            logging.info(f"len(id2seqs_{split} = {len(id2seqs_split)}")

        logging.info("Reading in labels ...")

        id2label_all = read_in_labels(path_to_file=label_path)
        id2label = {id: id2label_all[id] for id in id2seqs["all"].keys()}

        logging.info(f"len(id2label_all) = {len(id2label_all)}")
        logging.info(f"len(id2label) = {len(id2label)}")

        logging.info("Reading in embeddings ...")

        id2embedding = read_in_embeddings(path_to_file=embedding_paths["all"])

        if load_tmp_holdout_set:
            id2embedding = {**id2embedding, **read_in_embeddings(embedding_paths["holdout"])}

        embeddings = {key: id2embedding[key] for key in id2seqs["all"].keys()}
        logging.info(f"len(embeddings) = {len(embeddings)}")

        # remove duplicates and mismatched entries
        logging.info("Removing duplicates and mismatched entries ...")
        seq2count = collections.Counter(id2seqs["all"].values())
        for key_seq, value_count in seq2count.items():
            if value_count > 1:
                cath_ids = [
                    x for x, y in id2seqs["all"].items() if y == key_seq
                ]  # all IDs of the sequence

                # remove mismatched entries
                cath_labels = [id2label[cath_id] for cath_id in cath_ids]
                ids_to_remove = None
                if len(set(cath_labels)) > 1:
                    ids_to_remove = cath_ids
                elif without_duplicates:
                    ids_to_remove = cath_ids[1:]

                if ids_to_remove:
                    for cath_id in ids_to_remove:
                        for id2seqs_split in id2seqs.values():
                            id2seqs_split.pop(cath_id, None)
                        id2label.pop(cath_id, None)
                        id2embedding.pop(cath_id, None)

        standard_splits = [
            np.array([embeddings[prot_id] for prot_id in id2seqs["train"].keys()]),
            [id2label[prot_id] for prot_id in id2seqs["train"].keys()],
            np.array([embeddings[prot_id] for prot_id in id2seqs["val"].keys()]),
            [id2label[prot_id] for prot_id in id2seqs["val"].keys()],
            np.array([embeddings[prot_id] for prot_id in id2seqs["test"].keys()]),
            [id2label[prot_id] for prot_id in id2seqs["test"].keys()],
        ]

        holdout_data = (
            [
                np.array([embeddings[prot_id] for prot_id in id2seqs["holdout"].keys()]),
                [id2label[prot_id] for prot_id in id2seqs["holdout"].keys()],
            ]
            if load_tmp_holdout_set
            else [None, None]
        )

        lookup_data = (
            [
                np.array([embeddings[prot_id] for prot_id in id2seqs["lookup"].keys()]),
                [id2label[prot_id] for prot_id in id2seqs["lookup"].keys()],
            ]
            if load_lookup_set
            else [None, None]
        )

        dataset = Dataset(*standard_splits, *holdout_data, *lookup_data)

        if load_strings:
            dataset.load_strings(
                *[list(id2seqs_split.values()) for id2seqs_split in id2seqs.values()]
            )

        if shuffle_data:
            logging.info("Shuffling training set ...")
            dataset.shuffle_training_set(rng)

        if specific_level is not None and level_cutoff is None:
            for split in [
                dataset.y_train,
                dataset.y_val,
                dataset.y_test,
                dataset.y_tmp_holdout,
                dataset.y_lookup,
            ]:
                split = [label[specific_level] for label in split]
            dataset.train_labels = [label[specific_level] for label in dataset.train_labels]

        elif level_cutoff is not None and specific_level is None:
            for split in [
                dataset.y_train,
                dataset.y_val,
                dataset.y_test,
                dataset.y_tmp_holdout,
                dataset.y_lookup,
            ]:
                split = [label[:level_cutoff] for label in split]
            dataset.train_labels = [label[:level_cutoff] for label in dataset.train_labels]

        elif level_cutoff is not None and specific_level is not None:
            raise ValueError("Either specific_level or level_cutoff can be supplied, not both!")

        if encode_labels:
            dataset.encode_labels()

        logging.info("Serializing data for faster reloading ...")
        with open(data_dir / serialized_dataset_location, "wb+") as f:
            pickle.dump(dataset, f)

        return dataset

    def shape(self) -> Dict[str, Dict[str, Union[int, Tuple[int, int]]]]:
        return {
            "X": {
                "train": self.X_train.shape,
                "val": self.X_val.shape,
                "test": self.X_test.shape,
                "tmp_holdout": self.X_tmp_holdout.shape,
                "lookup": self.X_lookup.shape,
            },
            "y": {
                "train": len(self.y_train),
                "val": len(self.y_val),
                "test": len(self.y_test),
                "tmp_holdout": len(self.y_tmp_holdout),
                "lookup": len(self.y_lookup),
            },
        }

    def get_copy(self) -> Dataset:
        return deepcopy(self)

    def get_split(
        self,
        split: Literal["train", "val", "test", "tmp_holdout", "lookup"],
        x_encoding: Literal["embedding-array", "embedding-tensor", "string"] = "embedding-array",
        zipped: bool = False,
        y_encoding: Literal["list", "tensor"] = "list",
    ) -> Union[
        List[Tuple[np.ndarray, CATHLabel]],
        Tuple[np.ndarray, List[CATHLabel]],
        List[Tuple[torch.tensor, CATHLabel]],
        Tuple[torch.tensor, List[CATHLabel]],
        List[Tuple[str, CATHLabel]],
        Tuple[List[str], List[CATHLabel]],
    ]:
        x, y = {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
            "tmp_holdout": (self.X_tmp_holdout, self.y_tmp_holdout),
            "lookup": (self.X_lookup, self.y_lookup),
        }[split]

        if x_encoding == "embedding-tensor":
            # convert X embeddings to tensors
            x = torch.from_numpy(np.array(x))

        elif x_encoding == "string":
            if not self.stores_strings:
                raise ValueError(
                    "String representation requested, but no strings loaded. Use Dataset.load_strings() before calling get_split()"
                )
            x = {
                "train": self.X_train_str,
                "val": self.X_val_str,
                "test": self.X_test_str,
                "tmp_holdout": self.X_tmp_holdout_str,
                "lookup": self.X_lookup_str,
            }[split]

        if y_encoding == "tensor":
            y = torch.from_numpy(np.array(y, dtype=np.int64))

        if zipped:
            return list(zip(x, y))
        else:
            return x, y

    def get_filtered_version(self, cath_level: Literal["C", "A", "T", "H"]) -> Dataset:
        """
        Filter out all sequences from the validation & test set where there is no sequence sharing its CATH label
        up to the specified level
        """
        valid_labels = [label[:cath_level] for label in self.train_labels]

        def _filter_x_based_on_y(X, y):
            if X is None or y is None:
                return X, y
            x_filtered, y_filtered = [
                list(unzipped)
                for unzipped in zip(
                    *[
                        [embedding, label]
                        for embedding, label in zip(X, y)
                        if label[:cath_level] in valid_labels
                    ]
                )
            ]
            return x_filtered, y_filtered

        X_val, y_val = _filter_x_based_on_y(self.X_val, self.y_val)
        X_val = np.array(X_val)

        X_test, y_test = _filter_x_based_on_y(self.X_test, self.y_test)
        X_test = np.array(X_test)

        if self.X_tmp_holdout:
            X_tmp_holdout, y_tmp_holdout = _filter_x_based_on_y(
                self.X_tmp_holdout, self.y_tmp_holdout
            )
            X_tmp_holdout = np.array(X_tmp_holdout)
        else:
            X_tmp_holdout, y_tmp_holdout = None, None

        if self.X_lookup:
            X_lookup, y_lookup = _filter_x_based_on_y(self.X_lookup, self.y_lookup)
            X_lookup = np.array(X_lookup)
        else:
            X_lookup, y_lookup = None, None

        copy = Dataset(
            self.X_train,
            self.y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_tmp_holdout,
            y_tmp_holdout,
            X_lookup,
            y_lookup,
        )

        # TODO generalize
        if self.stores_strings:
            X_train_str, _ = _filter_x_based_on_y(self.X_train_str, self.y_train)
            X_val_str, _ = _filter_x_based_on_y(self.X_val_str, self.y_val)
            X_test_str, _ = _filter_x_based_on_y(self.X_test_str, self.y_test)
            X_tmp_holdout_str, _ = _filter_x_based_on_y(self.X_tmp_holdout_str, self.y_tmp_holdout)
            X_lookup_str, _ = _filter_x_based_on_y(self.X_lookup, self.y_lookup)
            copy.load_strings(X_train_str, X_val_str, X_test_str, X_tmp_holdout_str, X_lookup_str)

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
        self.X_tmp_holdout = scaler.transform(self.X_tmp_holdout)

    def load_strings(
        self,
        train: List[str],
        val: List[str],
        test: List[str],
        tmp_holdout: List[str],
        lookup: List[str],
    ) -> None:
        self.stores_strings = True
        self.X_train_str = train
        self.X_val_str = val
        self.X_test_str = test
        self.X_tmp_holdout_str = tmp_holdout
        self.X_lookup_str = lookup

    def encode_labels(self) -> None:
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([str(label) for label in self.all_labels])
        self.y_train = self.label_encoder.transform([str(label) for label in self.y_train])
        self.y_val = self.label_encoder.transform([str(label) for label in self.y_val])
        self.y_test = self.label_encoder.transform([str(label) for label in self.y_test])
        self.y_tmp_holdout = self.label_encoder.transform(
            [str(label) for label in self.y_tmp_holdout]
        )
        self.y_lookup = self.label_encoder.transform([str(label) for label in self.y_lookup])
