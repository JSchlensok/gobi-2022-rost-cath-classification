import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import pytorch_lightning as pl
import pytorch_metric_learning as pml
import torch
import torchmetrics as tm

from pytorch_metric_learning.losses import ArcFaceLoss, SubCenterArcFaceLoss
from sklearn.preprocessing import LabelEncoder

from gobi_cath_classification.pipeline import ModelInterface, load_data, Prediction
from .utils import get_base_dir

ROOT_DIR = get_base_dir()
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"


class ArcFaceModel(pl.LightningModule, ModelInterface):
    # TODO register tensors
    def __init__(
        self,
        config: Dict,
        lookup_data: Tuple[int, torch.utils.data.DataLoader],
        label_encoder: LabelEncoder,
        name: str,
        acceleration: str = "gpu",
        subcenters: bool = False,
    ):
        super().__init__()

        # Basic info
        self.name = name
        self.epoch = 0
        self.highest_acc = 0

        # Utilities
        self.acceleration = acceleration
        self.label_encoder = label_encoder
        self.pickle_intermediates = config["pickle_intermediates"]
        self.old_intermediate_path = None

        # Dataset
        self.num_classes, self.lookup_data = lookup_data
        self.lookup_labels = None
        self.lookup_embeddings = None
        self.query_embeddings = None

        # TODO initialize lookup labels here instead of reloading them every epoch

        # Hyperparameters
        self.model_lr = config["model_lr"]
        self.loss_lr = config["loss_lr"]
        if "layer_sizes" in config:
            self.l1_size, self.l2_size = config["layer_sizes"]
        else:
            self.l1_size, self.l2_size = 512, 128
        self.batch_size = config["batch_size"]

        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, self.l1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.l1_size, self.l2_size),
        )

        # https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AdaCos_Adaptively_Scaling_Cosine_Logits_for_Effectively_Learning_Deep_Face_CVPR_2019_paper.pdf
        scale = np.sqrt(2) * np.log(self.num_classes - 1)
        if not subcenters:
            self.loss_function = ArcFaceLoss(
                num_classes=self.num_classes, embedding_size=self.l2_size, margin=28.6, scale=scale
            )
        else:
            self.loss_function = SubCenterArcFaceLoss(
                num_classes=self.num_classes,
                embedding_size=self.l2_size,
                margin=28.6,  # default
                scale=scale,
                sub_centers=3,  # TODO tune?
            )
            # pass

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        model_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_lr)
        loss_optimizer = torch.optim.AdamW(self.loss_function.parameters(), lr=self.loss_lr)

        return [model_optimizer, loss_optimizer]

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

    # TODO type annotations
    # TODO error catching
    def predict_classes(self, query_projections):
        all_similarities = pml.distances.CosineSimilarity()(
            query_projections.float(), self.lookup_embeddings.float()
        )

        # https://stackoverflow.com/questions/56154604/groupby-aggregate-mean-in-pytorch
        M = torch.zeros(len(self.lookup_embeddings), self.num_classes)
        M[torch.arange(len(self.lookup_embeddings)), self.lookup_labels] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=0)

        if self.acceleration == "gpu":
            M = M.to(self.device).cuda()

        class_similarities = torch.mm(all_similarities, M)

        return class_similarities

    def validation_step(self, batch, batch_idx):
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

        # Validation accuracy
        class_similarities = self.predict_classes(query_projections)
        # Move to CPU for torchmetrics
        acc = tm.Accuracy()(class_similarities.to("cpu"), y.to("cpu"))
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        # TODO move to proper callback
        if self.pickle_intermediate and acc > self.highest_acc:
            self.highest_acc = acc
            path = ROOT_DIR / f"models/{self.name}_val_acc_{acc:.2f}.pickle"
            print(f"Pickling to {path} ...")
            with open(path, "wb+") as f:
                pickle.dump(self, f)

            if self.old_path:
                os.remove(self.old_path)
            self.old_path = path

        for k in [5, 10, 25, 50, 100]:
            self.log(
                f"val_acc_top{k}",
                tm.Accuracy(top_k=k)(class_similarities.to("cpu"), y.to("cpu")),
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        # - ignore_index?
        x, y = batch
        if self.acceleration == "cpu":
            # No automatic mixed precision possible
            x = x.type(torch.float32)

        query_projections = self(x)

        test_loss = self.loss_function(query_projections, y)
        self.log("test_loss", test_loss)

        class_similarities = self.predict_classes(query_projections)

        # TODO merge top k predictions
        # Move to CPU for torchmetrics
        self.log("test_acc", tm.Accuracy()(class_similarities.to("cpu"), y.to("cpu")))

    def predict_step(self, batch, batch_idx):
        x, y = batch
        if self.acceleration == "cpu":
            # No automatic mixed precision possible
            x = x.type(torch.float32)
        query_projections = self(x)

        class_similarities = self.predict_classes(query_projections)
        _, neighbor_indices = torch.topk(class_similarities, 1, largest=True)
        encoded_labels = torch.take(self.lookup_labels, neighbor_indices)
        decoded_labels = self.label_encoder.inverse_transform(encoded_labels.cpu().flatten())

        return decoded_labels

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        # TODO does this make sense?
        # - just run training_step ?
        raise NotImplementedError

    # TODO enable different lookup file than training set
    # TODO use self.predict_step() ?
    def predict(self, embeddings) -> Prediction:
        # manual casting required since this is outside of what PyTorch lightning takes care of
        if self.acceleration == "gpu":
            embeddings = embeddings.type(torch.float32).cuda()
            query_projections = self(embeddings).to(self.device).cuda()

        class_similarities = self.predict_classes(query_projections)
        _, neighbor_indices = torch.topk(class_similarities, k=1, largest=True)
        encoded_labels = torch.take(self.lookup_labels, neighbor_indices)
        pred_labels = self.label_encoder.inverse_transform(encoded_labels.cpu().flatten())

        df = pd.DataFrame(
            data=class_similarities.cpu().detach().numpy(),
            columns=self.label_encoder.inverse_transform(np.arange(self.num_classes)),
        )

        # sort out negative similarities
        df[df < 0] = 0

        return Prediction(df)

    def save_checkpoint(self, save_to_dir: Path):
        filename = self.name + f"_epoch_{self.epoch}.pth"
        torch.save(self.model.fnn.state_dict(), save_to_dir / filename)
