from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from torch import nn
from torch.nn.functional import one_hot

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline import torch_utils


class RandomForestModel(ModelInterface):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        class_weight=None,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            class_weight=class_weight,
        )

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> None:
        self.model.fit(X=embeddings, y=labels, sample_weight=sample_weights)

    def predict_proba(self, embeddings: np.ndarray) -> Prediction:
        predictions = self.model.predict_proba(X=embeddings)
        df = pd.DataFrame(data=predictions, columns=self.model.classes_)
        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError


class GaussianNaiveBayesModel(ModelInterface):
    def __init__(self):
        self.model = GaussianNB()

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> None:
        self.model.fit(X=embeddings, y=labels, sample_weight=sample_weights)

    def predict_proba(self, embeddings: np.ndarray) -> Prediction:
        predictions = self.model.predict_proba(X=embeddings)
        df = pd.DataFrame(data=predictions, columns=self.model.classes_)
        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError


class NeuralNetworkModel(ModelInterface):
    def __init__(
        self,
        lr: float,
        class_names: List[str],
        layer_sizes: List[int],
        batch_size: int,
        optimizer: str,
    ):
        self.device = torch_utils.get_device()

        self.batch_size = batch_size
        self.class_names = sorted(class_names)
        model = nn.Sequential()

        for i, num_in_features in enumerate(layer_sizes[:-1]):
            model.add_module(
                f"Linear_{i}",
                nn.Linear(
                    in_features=num_in_features,
                    out_features=layer_sizes[i + 1],
                ),
            )
            model.add_module(f"ReLU_{i}", nn.ReLU())

        model.add_module(
            f"Linear_{len(layer_sizes) - 1}",
            nn.Linear(in_features=layer_sizes[-1], out_features=len(self.class_names)).to(
                self.device
            ),
        )

        model.add_module("Softmax", nn.Softmax())
        self.model = model.to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer is not valid: {optimizer}")

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> None:

        permutation = torch.randperm(len(embeddings_tensor))
        X = embeddings_tensor.to(self.device)
        y_indices = torch.tensor([self.class_names.index(label) for label in labels]).to(
            self.device
        )
        y_one_hot = 1.0 * one_hot(y_indices, num_classes=len(self.class_names))

        for i in range(0, len(embeddings), self.batch_size):
            self.optimizer.zero_grad()
            indices = permutation[i : i + self.batch_size]
            batch_X = X[indices].float()
            batch_y = y_one_hot[indices]
            y_pred = self.model(batch_X)
            loss = self.loss_function(y_pred, batch_y)
            loss.backward()
            self.optimizer.step()

    def predict_proba(self, embeddings: np.ndarray) -> Prediction:
        pred_proba = self.model(torch.from_numpy(embeddings).float().to(self.device))
        df = pd.DataFrame(pred_proba, columns=self.class_names).astype("float")
        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError
