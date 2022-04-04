import numpy as np
from pathlib import Path

from ..data import load_data


def test_loading_with_shuffling():
    dataset = load_data(
        data_dir=Path(__file__).parent.parent.parent.parent.parent / "data",
        rng=np.random.RandomState(42),
        without_duplicates=True,
        shuffle_data=True,
        load_only_small_sample=False,
        reloading_allowed=True,
        load_tmp_holdout_set=True,
    )


def test_loading_without_shuffling():
    dataset = load_data(
        data_dir=Path(__file__).parent.parent.parent.parent.parent / "data",
        rng=np.random.RandomState(42),
        without_duplicates=True,
        shuffle_data=True,
        load_only_small_sample=False,
        reloading_allowed=True,
        load_tmp_holdout_set=True,
    )


def test_loading_with_strings():
    dataset = load_data(
        data_dir=Path(__file__).parent.parent.parent.parent.parent / "data",
        rng=np.random.RandomState(42),
        without_duplicates=True,
        shuffle_data=True,
        load_only_small_sample=False,
        reloading_allowed=True,
        load_strings=True,
        load_tmp_holdout_set=True,
    )
