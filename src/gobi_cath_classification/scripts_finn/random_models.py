from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import torch
from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import set_random_seeds


class RandomBaseline1(ModelInterface):
    def __init__(
            self,
            class_balance: bool,
            rng: np.random.RandomState,
            random_seed: int = 42
    ):
        self.device = torch_utils.get_device()

        self.random_seed = random_seed
        self.rng = rng
        print(f"rng = {rng}")
        set_random_seeds(seed=random_seed)

        self.class_balance = class_balance

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        pass

    def predict(self, embeddings: np.ndarray) -> Prediction:
        pass


    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError