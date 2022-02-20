import os
from pathlib import Path

from ..data import Dataset, load_data


def test_loading():
    dataset = load_data(
        Path(__file__).parent.parent.parent.parent.parent / "data", True, True, False
    )
