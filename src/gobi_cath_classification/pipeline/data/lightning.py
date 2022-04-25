from typing import Optional
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch

from gobi_cath_classification.pipeline import Dataset, load_data
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir

ROOT_DIR = get_base_dir()
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = DATA_DIR, batch_size: int = 32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        self.x = {}
        self.y = {}
        self.dataset = None
        self.num_classes = None

        self.label_encoder = None

    def setup(self, stage: Optional[str] = None) -> None:

        dataset = load_data(
            self.data_dir,
            rng=np.random.RandomState(42),
            without_duplicates=True,
            shuffle_data=False,
            load_only_small_sample=False,
            reloading_allowed=True,
            load_tmp_holdout_set=True,
            load_lookup_set=True,
            encode_labels=True,
        )

        self.num_classes = len(dataset.all_labels)

        self.label_encoder = dataset.label_encoder

        for split in ["train", "val", "test", "tmp_holdout", "lookup"]:
            self.x[split], self.y[split] = dataset.get_split(
                split, x_encoding="embedding-tensor", zipped=False, y_encoding="tensor"
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x["train"], self.y["train"]),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x["val"], self.y["val"]),
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x["test"], self.y["test"]),
            batch_size=self.batch_size,
        )

    def lookup_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x["lookup"], self.y["lookup"]),
            batch_size=self.batch_size,
        )
