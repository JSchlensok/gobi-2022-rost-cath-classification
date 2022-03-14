from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

from gobi_cath_classification.pipeline.prediction import Prediction


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
