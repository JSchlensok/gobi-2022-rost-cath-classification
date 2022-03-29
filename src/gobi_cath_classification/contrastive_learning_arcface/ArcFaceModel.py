import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.distributions.empirical_distribution import ECDF

from gobi_cath_classification.pipeline import ModelInterface, load_data
from gobi_cath_classification.pipeline import Prediction
from .eat import EAT
from .utils import get_base_dir

ROOT_DIR = get_base_dir()
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"


class ArcFaceModel(pl.LightningModule, ModelInterface):
    # TODO set hyperparameters
    def __init__(
        self,
        lookup_data: Tuple[int, torch.utils.data.DataLoader],
        label_encoder: LabelEncoder,
        accuracy_calculator: AccuracyCalculator,
        name: str,
        lr: float = 1e-2,
        acceleration: str = "gpu",
    ):
        super().__init__()
        self.num_classes, self.lookup_data = lookup_data
        self.name = name
        self.epoch = 0
        self.lr = lr
        self.acceleration = acceleration
        self.label_encoder = label_encoder
        self.accuracy_calculator = accuracy_calculator

        self.lookup_labels = None
        self.lookup_embeddings = None
        self.query_embeddings = None

        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128)
        )  # TODO tune
        self.loss_function = ArcFaceLoss(
            num_classes=self.num_classes, embedding_size=128
        )  # TODO tune to also allow SubCenterArcFaceLoss
        # TODO dynamically set embedding_size

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # TODO tune optimizer choices & learning rates
        model_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_optimizer = torch.optim.AdamW(
            self.loss_function.parameters(), lr=self.lr / 100
        )  # 1/100 ratio from example at https://colab.research.google.com/github/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/SubCenterArcFaceMNIST.ipynb

        return [model_optimizer, loss_optimizer]

        # TODO tune with cyclic & 1cycle as choices
        """
        model_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer)
        loss_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(loss_optimizer)
        return [model_optimizer, loss_optimizer], {
            "scheduler": model_lr_scheduler,
            "monitor": "val_loss",
        }
        """

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        if self.acceleration == "cpu":
            # No automatic mixed precision possible
            x = x.type(torch.float32)
        projections = self(x)

        self.lookup_embeddings = torch.cat([self.lookup_embeddings, projections])
        self.lookup_labels = torch.cat([self.lookup_labels, y])

        loss = self.loss_function(projections, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def on_train_epoch_start(self) -> None:
        # Clear embedding cache
        self.lookup_embeddings = torch.Tensor().to(self.device)
        self.lookup_labels = torch.Tensor().to(self.device).type(torch.int64)
        self.query_embeddings = torch.Tensor().to(self.device)

    def validation_step(self, batch, batch_idx):
        # TODO top-k metrics
        # TODO per-level metrics
        # compute embeddings for validation batch
        x, y = batch
        if self.acceleration == "cpu":
            # No automatic mixed precision possible
            x = x.type(torch.float32)
        query_projections = self(x)

        # Validation loss
        loss = self.loss_function(query_projections, y)
        self.log("val_loss", loss, prog_bar=True)

        eat = EAT(CosineSimilarity(), self.lookup_embeddings, query_projections)
        eat.get_neighbors(1)
        eat.transfer_labels(self.lookup_labels)
        eat.decode_labels(self.label_encoder)

        predicted_labels = eat.decoded_labels
        true_labels = self.label_encoder.inverse_transform(y.cpu().flatten())

        # Compute accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        self.log("val_acc", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_function(y_hat, y)
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        # TODO
        pass

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        # TODO does this make sense?
        raise NotImplementedError

    def predict(self, embeddings: np.ndarray) -> Prediction:
        # TODO enable different lookup file than training set
        # TODO get probabilities differently
        #   - as inverse of softmax loss?
        #   - from cosine similarity?
        logging.info("Loading dataset")
        dataset = load_data(DATA_DIR, np.random.RandomState(42), True, True, False, True)

        logging.info("Getting splits from dataset")
        lookup_embeddings, lookup_labels = dataset.get_split(
            "train", x_encoding="embedding-tensor", zipped=False
        )

        query_embeddings, query_labels = dataset.get_split(
            "test", x_encoding="embedding-tensor", zipped=False
        )

        lookup_embeddings = lookup_embeddings.cuda()
        query_embeddings = query_embeddings.cuda()

        with torch.cuda.amp.autocast():
            lookup_embeddings = self.model(lookup_embeddings)
            query_embeddings = self.model(query_embeddings)

        logging.info("Transferring annotations ...")

        all_distances = np.empty((query_embeddings.shape[0], len(lookup_labels)))

        for i, query_embedding in enumerate(query_embeddings):
            distances = torch.linalg.norm(
                lookup_embeddings.float() - query_embedding.float().unsqueeze(dim=0), dim=1
            )
            all_distances[i] = distances.cpu().detach().numpy()

        # TODO change to use flag for triggering this
        if True:
            normed = np.zeros(all_distances.shape)
            normed[np.arange(len(all_distances)), all_distances.argmax(1)] = 1
            df = pd.DataFrame(data=normed, columns=[str(label) for label in lookup_labels])
            df = df.groupby(df.columns, axis=1).max()

        else:
            ecdf = ECDF(all_distances.ravel())

            # compute prediction certainty from CDF
            logging.info("Computing probabilities from distances ...")
            probabilities = ecdf(all_distances)

            """
            logging.debug("Evaluating predictions")
            str_preds = [str(pred) for pred in predictions]
            for level in "CATH":
                print(level)
                print(accuracy_for_level(query_labels, str_preds, dataset.train_labels, level))
            """
            logging.info("Creating Dataframe ...")
            df = pd.DataFrame(data=probabilities, columns=[str(label) for label in lookup_labels])
            logging.info("Sorting Dataframe ...")
            df = df.groupby(df.columns, axis=1).mean()

        return Prediction(df)

    def save_checkpoint(self, save_to_dir: Path):
        filename = self.name + f"_epoch_{self.epoch}.pth"
        torch.save(self.model.fnn.state_dict(), save_to_dir / filename)
