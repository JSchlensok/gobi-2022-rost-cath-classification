from pathlib import Path
from collections import Counter
from typing import Dict
from typing_extensions import Literal

import h5py
import numpy as np
import pandas as pd
import pickle

from gobi_cath_classification.pipeline.utils.CATHLabel import CATHLabel
from gobi_cath_classification.pipeline.data import Dataset
from gobi_cath_classification.pipeline.utils.torch_utils import RANDOM_SEED

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"


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


def read_in_labels(path_to_file: Path) -> Dict[str, CATHLabel]:
    df = pd.read_csv(filepath_or_buffer=path_to_file, delimiter=r"\s+", header=None, comment="#")

    id2label = {
        row[0]: CATHLabel(".".join([str(level) for level in row[1:5]])) for _, row in df.iterrows()
    }

    return id2label


def load_data(
    data_dir: Path,
    rng: np.random.RandomState,
    without_duplicates: bool,
    load_tmp_holdout_set: bool,
    load_strings: bool = False,
    load_only_small_sample: bool = False,
    shuffle_data: bool = True,
    reloading_allowed: bool = False,
    # Load data with Y-values only being one level
    specific_level: Literal["C", "A", "T", "H"] = None,
    # Load data with Y-values only up to the given cutoff level
    level_cutoff: Literal["C", "A", "T", "H"] = None,
):
    print(f"Loading data from directory: {data_dir}")

    path_sequences_train = data_dir / "train74k.fasta"
    path_sequences_val = data_dir / "val200.fasta"
    path_sequences_test = data_dir / "test219.fasta"
    path_sequences_tmp_holdout = data_dir / "holdout389.fasta"

    path_embeddings = data_dir / "cath_v430_dom_seqs_S100_161121.h5"
    path_labels = data_dir / "cath-domain-list.txt"

    if load_tmp_holdout_set:
        path_labels = data_dir / "cath-domain-list-updated.txt"
        path_embeddings_tmp_holdout = data_dir / "temporal_holdout_set.h5"

    if load_only_small_sample:
        path_sequences_train = data_dir / "sample_data/sample_train100.fasta"
        path_labels = data_dir / "sample_data/sample_cath-domain-list100.txt"

    # Reload if possible
    duplicates_tag = "no-duplicates" if without_duplicates else "duplicates"
    small_sample_tag = "small_sample" if load_only_small_sample else "full"
    tmp_holdout_tag = "with_tmp_holdout" if load_tmp_holdout_set else "no-tmp_holdout"
    including_strings_tag = "_with_strings" if load_strings else ""
    serialized_dataset_location = f"serialized_dataset_{duplicates_tag}_{small_sample_tag}_{tmp_holdout_tag}{including_strings_tag}.pickle"

    if reloading_allowed:
        print("Trying to find a serialized dataset ...")

        if (data_dir / serialized_dataset_location).exists():
            print("Found a serialized dataset to save time")
            with open(data_dir / serialized_dataset_location, "rb") as f:
                serialized_dataset = pickle.load(f)

            if shuffle_data:
                serialized_dataset.shuffle_training_set(np.random.RandomState(RANDOM_SEED))

            return serialized_dataset

        else:
            print("Couldn't find a matching serialized dataset, creating a new one")

    print("Reading in sequences ...")
    id2seqs_train = read_in_sequences(path_sequences_train)
    id2seqs_val = read_in_sequences(path_sequences_val)
    id2seqs_test = read_in_sequences(path_sequences_test)
    id2seqs_tmp_holdout = {}

    id2seqs_all = {**id2seqs_train, **id2seqs_val, **id2seqs_test}
    if load_tmp_holdout_set:
        id2seqs_tmp_holdout = read_in_sequences(path_sequences_tmp_holdout)
        id2seqs_all = {**id2seqs_all, **id2seqs_tmp_holdout}

    print(f"len(id2seqs_train) = {len(id2seqs_train)}")
    print(f"len(id2seqs_val) = {len(id2seqs_val)}")
    print(f"len(id2seqs_test) = {len(id2seqs_test)}")
    print(f"len(id2seqs_tmp_holdout) = {len(id2seqs_tmp_holdout)}")
    print(f"len(id2seqs_all = {len(id2seqs_all)}")

    print("Reading in labels ...")
    id2label_all = read_in_labels(path_to_file=path_labels)
    id2label = {key: id2label_all[key] for key in id2seqs_all.keys()}

    print(f"len(id2label_all) = {len(id2label_all)}")
    print(f"len(id2label) = {len(id2label)}")

    print("Reading in embeddings ...")
    if load_tmp_holdout_set:
        id2emb_tmp_holdout = read_in_embeddings(path_to_file=path_embeddings_tmp_holdout)
        id2embedding = {**read_in_embeddings(path_to_file=path_embeddings), **id2emb_tmp_holdout}
    else:
        id2embedding = read_in_embeddings(path_to_file=path_embeddings)
    embeddings = {key: id2embedding[key] for key in id2seqs_all.keys()}
    print(f"len(embeddings) = {len(embeddings)}")

    # remove duplicates and mismatched entries
    print("Removing duplicates and mismatched entries ...")
    seq2count = Counter(id2seqs_all.values())
    for key_seq, value_count in seq2count.items():
        if value_count > 1:
            cath_ids = [
                x for x, y in id2seqs_all.items() if y == key_seq
            ]  # all IDs of the sequence

            # remove mismatched entries
            cath_labels = [id2label[cath_id] for cath_id in cath_ids]
            if len(set(cath_labels)) > 1:
                for cath_id in cath_ids:
                    id2seqs_train.pop(cath_id, None)
                    id2seqs_val.pop(cath_id, None)
                    id2seqs_test.pop(cath_id, None)
                    id2seqs_tmp_holdout.pop(cath_id, None)
                    id2seqs_all.pop(cath_id, None)
                    id2label.pop(cath_id, None)
                    id2embedding.pop(cath_id, None)

            # remove duplicates
            elif without_duplicates:
                for cath_id in cath_ids[1:]:
                    id2seqs_train.pop(cath_id, None)
                    id2seqs_val.pop(cath_id, None)
                    id2seqs_test.pop(cath_id, None)
                    id2seqs_tmp_holdout.pop(cath_id, None)
                    id2seqs_all.pop(cath_id, None)
                    id2label.pop(cath_id, None)
                    id2embedding.pop(cath_id, None)

    dataset = Dataset(
        X_train=np.array([embeddings[prot_id] for prot_id in id2seqs_train.keys()]),
        y_train=[id2label[prot_id] for prot_id in id2seqs_train.keys()],
        train_labels=[id2label[prot_id] for prot_id in id2seqs_train.keys()],
        X_val=np.array([embeddings[prot_id] for prot_id in id2seqs_val.keys()]),
        y_val=[id2label[prot_id] for prot_id in id2seqs_val.keys()],
        X_test=np.array([embeddings[prot_id] for prot_id in id2seqs_test.keys()]),
        y_test=[id2label[prot_id] for prot_id in id2seqs_test.keys()],
        X_tmp_holdout=np.array([embeddings[prot_id] for prot_id in id2seqs_tmp_holdout.keys()]),
        y_tmp_holdout=([id2label[prot_id] for prot_id in id2seqs_tmp_holdout.keys()]),
    )

    if load_strings:
        dataset.load_strings(
            list(id2seqs_train.values()),
            list(id2seqs_val.values()),
            list(id2seqs_test.values()),
            list(id2seqs_tmp_holdout.values()),
        )

    if shuffle_data:
        print("Shuffling training set ...")
        dataset.shuffle_training_set(rng)

    if specific_level is not None and level_cutoff is None:
        dataset.y_train = [label[specific_level] for label in dataset.y_train]
        dataset.train_labels = [label[specific_level] for label in dataset.train_labels]
        dataset.y_val = [label[specific_level] for label in dataset.y_val]
        dataset.y_test = [label[specific_level] for label in dataset.y_test]
        dataset.y_tmp_holdout = [label[specific_level] for label in dataset.y_tmp_holdout]
    elif level_cutoff is not None and specific_level is None:
        dataset.y_train = [label[:level_cutoff] for label in dataset.y_train]
        dataset.train_labels = [label[:level_cutoff] for label in dataset.train_labels]
        dataset.y_val = [label[:level_cutoff] for label in dataset.y_val]
        dataset.y_test = [label[:level_cutoff] for label in dataset.y_test]
        dataset.y_tmp_holdout = [label[:level_cutoff] for label in dataset.y_tmp_holdout]
    elif level_cutoff is not None and specific_level is not None:
        raise ValueError("Either specific_level or level_cutoff can be supplied, not both!")

    print("Serializing data for faster reloading ...")
    with open(data_dir / serialized_dataset_location, "wb+") as f:
        pickle.dump(dataset, f)

    return dataset
