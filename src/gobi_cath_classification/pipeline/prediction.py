import warnings

import numpy as np
from typing import List

import pandas as pd
from pathlib import Path


class Prediction:
    def __init__(self, probabilities: pd.DataFrame, ids: List[str] = None):
        self.probabilities = probabilities
        self.ids = ids
        assert list(probabilities.columns) == sorted(probabilities.columns)
        for col in probabilities.columns:
            assert (
                type(col) == str
            ), f"Your column ({col}) should be a string, but it is of type: {type(col)}"
            if len(col.split(".")) != 4:
                warnings.warn("Predictions do not cover all levels. Check if this is intended!")

    def argmax_labels(self) -> List[str]:
        y_pred_argmax_val = np.argmax(self.probabilities.values, axis=1)
        y_pred_strings_val = [self.probabilities.columns[y] for y in y_pred_argmax_val]
        return y_pred_strings_val


def save_predictions(pred: Prediction, directory: Path, filename: str) -> None:
    with open(Path(directory / filename), "w") as f:
        column_names = "\t".join(pred.probabilities.columns)
        f.write(f"{column_names}\n")
        for probas in pred.probabilities.values:
            p = "\t".join([str(proba) for proba in probas])
            f.write(f"{p}\n")


def read_in_predictions(filepath: Path) -> Prediction:
    df = pd.read_csv(filepath_or_buffer=filepath, sep="\t")
    return Prediction(probabilities=df)
