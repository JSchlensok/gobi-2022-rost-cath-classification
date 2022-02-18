from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import torch
from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import set_random_seeds
from gobi_cath_classification.pipeline.data_loading import DataSplits


class RandomBaseline(ModelInterface):
    """
    The RandomBaseline model does not need the train_one_epoch function
    it only predicts a random class for each input of the predict method
    to predict a random class, we implement two different methods:
        1. The class sizes are ignored and we just generate random numbers
        2. The class sizes are taken into account during the prediction generation
    We therefore have the parameter class_balance to differ between these two methods
    """

    def __init__(

            self,
            data: DataSplits,
            class_balance: bool,
            rng: np.random.RandomState,
            random_seed: int = 42,

    ):
        """
        Args:
            data: a DataSplits object created from the data in the data folder
            class_balance: differentiate between the two methods mentioned above
            rng: random seed setting
            random_seed: random seed setting
        """
        self.data = data
        self.class_balance = class_balance
        self.device = torch_utils.get_device()
        self.random_seed = random_seed
        self.rng = rng
        print(f"rng = {rng}")
        set_random_seeds(seed=random_seed)

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        pass

    def predict(self, embeddings: np.ndarray) -> Prediction:
        """

        Args:
            embeddings: the input embeddings from which we want to generate a random prediction

        Returns: a Prediction object (panda df) containing for each column a predicted value and each row
        is a element from the input embeddings

        """
        self.data.all_labels_train_sorted
        pass


    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError