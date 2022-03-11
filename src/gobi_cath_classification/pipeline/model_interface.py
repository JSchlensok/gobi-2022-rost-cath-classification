import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch


class Prediction:
    def __init__(self, probabilities: pd.DataFrame, ids: List[str] = None):
        self.probabilities = probabilities
        self.ids = ids
        assert list(probabilities.columns) == sorted(probabilities.columns)
        for col in probabilities.columns:
            assert (
                type(col) == str
            ), f"Your column ({col}) should be a string, but it is of type: {type(col)}"
            assert len(col.split(".")) == 4

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


class ModelInterface:
    @abstractmethod
    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """
        Trains the model.

        Args:
            embeddings:
                2D array with shape (number of embeddings, 1024)
            embeddings_tensor:
                torch.Tensor(embeddings) for torch models
            labels:
                2D array with shape (number of embeddings, number of classes) of
                one-hot-encoded labels
            sample_weights:
                1D array with sample weights, shape (number of embeddings)
        Returns:
            dictionary with model-specific metrics
        """

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> Prediction:
        """
        Predicts probabilities for the CATH superfamily labels.

        Args:
            embeddings:
                2D array with shape (number of embeddings, 1024)

        Returns:
            Pandas DataFrame with shape (number of embeddings, number of classes) of
            probabilities. Each column corresponds to one CATH superfamily.
        """

    @abstractmethod
    def save_checkpoint(self, save_to_dir: Path):
        """

        Save a checkpoint to given directory.

        """

    @abstractmethod
    def load_model_from_checkpoint(self, checkpoint_file_dir: Path):
        """

        Load model from given checkpoint file(s),

        """
