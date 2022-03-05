from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import one_hot

from gobi_cath_classification.pipeline.data_loading import (
    read_in_sequences,
    read_in_labels,
    DataSplits,
)

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"


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

            # remove duplicates
            elif without_duplicates:
                for cath_id in cath_ids[1:]:
                    id2seqs_train.pop(cath_id, None)
                    id2seqs_val.pop(cath_id, None)
                    id2seqs_test.pop(cath_id, None)
                    id2seqs_all.pop(cath_id, None)
                    id2label.pop(cath_id, None)

    dataset = DataSplits(
        X_train=np.array([id2seqs_all[cath_id] for cath_id in id2seqs_train.keys()]),
        y_train=[id2label[cath_id] for cath_id in id2seqs_train.keys()],
        X_val=np.array([id2seqs_all[cath_id] for cath_id in id2seqs_val.keys()]),
        y_val=[id2label[cath_id] for cath_id in id2seqs_val.keys()],
        X_test=np.array([id2seqs_all[cath_id] for cath_id in id2seqs_test.keys()]),
        y_test=[id2label[cath_id] for cath_id in id2seqs_test.keys()],
        all_labels_train_sorted=sorted(list(set([id2label[k] for k in id2seqs_train.keys()]))),
    )

    if shuffle_data:
        return dataset.shuffled(rng=rng)
    else:
        return dataset
