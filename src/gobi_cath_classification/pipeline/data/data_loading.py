import pickle
from collections import Counter
from pathlib import Path
from typing import Dict

from deprecation import deprecated
import h5py
import numpy as np
import pandas as pd
from typing_extensions import Literal

from gobi_cath_classification.pipeline.data import Dataset
from gobi_cath_classification.pipeline.utils.CATHLabel import CATHLabel
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


@deprecated(details="Use class method Dataset.load() instead")
def load_data(
    data_dir: Path,
    rng: np.random.RandomState,
    without_duplicates: bool,
    load_tmp_holdout_set: bool,
    load_lookup_set: bool = False,
    load_strings: bool = False,
    load_only_small_sample: bool = False,
    shuffle_data: bool = True,
    reloading_allowed: bool = False,
    # Load data with Y-values only being one level
    specific_level: Literal["C", "A", "T", "H"] = None,
    # Load data with Y-values only up to the given cutoff level
    level_cutoff: Literal["C", "A", "T", "H"] = None,
    encode_labels: bool = False,
):
    return Dataset.load(
        data_dir,
        rng,
        without_duplicates,
        load_tmp_holdout_set,
        load_lookup_set,
        load_strings,
        load_only_small_sample,
        shuffle_data,
        reloading_allowed,
        specific_level,
        level_cutoff,
        encode_labels,
    )
