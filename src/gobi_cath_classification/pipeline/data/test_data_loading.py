import numpy as np
from pathlib import Path

from ..data import Dataset, load_data


def test_loading_with_shuffling():
    dataset = load_data(
        Path(__file__).parent.parent.parent.parent.parent / "data", np.random.RandomState(42), True, True, False, True
    )

def test_loading_without_shuffling():
    dataset = load_data(
        Path(__file__).parent.parent.parent.parent.parent / "data", np.random.RandomState(42), True, True, False, True
    )
