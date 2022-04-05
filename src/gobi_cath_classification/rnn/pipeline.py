import pickle
from collections import Counter
import h5py
from pathlib import Path
from gobi_cath_classification.pipeline.data.data_loading import (
    read_in_embeddings,
    read_in_sequences,
    read_in_labels,
)
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gobi_cath_classification.pipeline.utils.torch_utils import RANDOM_SEED

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"


def load_data(
    data_dir: Path,
    without_duplicates: bool = True,
    load_only_small_sample: bool = False,
    load_tmp_holdout_set: bool = False
):
    print(f"Loading data from directory: {data_dir}")

    path_sequences_train = data_dir / "train74k.fasta"
    path_sequences_val = data_dir / "val200.fasta"
    path_sequences_test = data_dir / "test219.fasta"
    path_sequences_tmp_holdout = data_dir / "holdout389.fasta"

    path_embeddings = data_dir / "t5_xl_v3_half_cath_S100_RESIDUE.h5"
    path_labels = data_dir / "cath-domain-list.txt"

    if load_tmp_holdout_set:
        path_labels = data_dir / "cath-domain-list-updated.txt"
        path_embeddings_tmp_holdout = data_dir / "temporal_holdout_set.h5"

    if load_only_small_sample:
        path_sequences_train = data_dir / "sample_data/sample_train100.fasta"
        path_labels = data_dir / "sample_data/sample_cath-domain-list100.txt"

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
        id2emb_tmp_holdout = read_in_embeddings(path_to_file=path_embeddings_tmp_holdout, save_ram=True)
        id2embedding = {**read_in_embeddings(path_to_file=path_embeddings, save_ram=True), **id2emb_tmp_holdout}
    else:
        id2embedding = read_in_embeddings(path_to_file=path_embeddings, save_ram=True)
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

    return_tuple = [
        [embeddings[prot_id] for prot_id in id2seqs_train.keys()],
        [id2label[prot_id] for prot_id in id2seqs_train.keys()],
        [id2label[prot_id] for prot_id in id2seqs_train.keys()],
        [embeddings[prot_id] for prot_id in id2seqs_val.keys()],
        [id2label[prot_id] for prot_id in id2seqs_val.keys()],
        [embeddings[prot_id] for prot_id in id2seqs_test.keys()],
        [id2label[prot_id] for prot_id in id2seqs_test.keys()],
    ]

    if load_tmp_holdout_set:
        return_tuple.append([embeddings[prot_id] for prot_id in id2seqs_tmp_holdout.keys()])
        return_tuple.append([id2label[prot_id] for prot_id in id2seqs_tmp_holdout.keys()])

    return return_tuple
