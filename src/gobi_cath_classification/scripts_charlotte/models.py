import math
import sys
from pathlib import Path
from typing import List, Optional, Dict
from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from torch import nn
from torch.nn.functional import one_hot

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import set_random_seeds


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
    ) -> Dict[str, float]:
        self.model.fit(X=embeddings, y=labels, sample_weight=sample_weights)
        model_specific_metrics = {}
        return model_specific_metrics

    def predict(self, embeddings: np.ndarray) -> Prediction:
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
    ) -> Dict[str, float]:
        self.model.fit(X=embeddings, y=labels, sample_weight=sample_weights)
        model_specific_metrics = {}
        return model_specific_metrics

    def predict(self, embeddings: np.ndarray) -> Prediction:
        predictions_proba = self.model.predict_proba(X=embeddings)
        df = pd.DataFrame(data=predictions_proba, columns=self.model.classes_)
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
        dropout_sizes: List[Optional[float]],
        batch_size: int,
        optimizer: str,
        loss_function: Literal["CrossEntropyLoss", "HierarchicalLoss"],
        class_weights: torch.Tensor,
        rng: np.random.RandomState,
        random_seed: int = 42,
        weight_decay: float = 0.0,
    ):
        assert len(layer_sizes) == len(dropout_sizes)
        self.device = torch_utils.get_device()

        self.random_seed = random_seed
        self.rng = rng
        print(f"rng = {rng}")
        set_random_seeds(seed=random_seed)

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
            if dropout_sizes[i] is not None:
                model.add_module(
                    f"Dropout_{i}",
                    nn.Dropout(
                        p=dropout_sizes[i],
                    ),
                )
            model.add_module(f"ReLU_{i}", nn.ReLU())

        model.add_module(
            f"Linear_{len(layer_sizes) - 1}",
            nn.Linear(in_features=layer_sizes[-1], out_features=len(self.class_names)).to(
                self.device
            ),
        )
        if dropout_sizes[-1] is not None:
            model.add_module(
                f"Dropout_{len(dropout_sizes)-1}",
                nn.Dropout(
                    p=dropout_sizes[-1],
                ),
            )

        # model.add_module("Softmax", nn.Softmax())
        self.model = model.to(self.device)
        if loss_function == "CrossEntropyLoss":
            self.loss_function = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None,
            )
        elif loss_function == "HierarchicalLoss":
            self.loss_function = hierarchical_loss
        else:
            raise ValueError(f"Loss_function is not valid: {loss_function}")

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer is not valid: {optimizer}")

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:

        permutation = torch.randperm(len(embeddings_tensor))
        X = embeddings_tensor.to(self.device)
        y_indices = torch.tensor([self.class_names.index(label) for label in labels]).to(
            self.device
        )
        y_one_hot = 1.0 * one_hot(y_indices, num_classes=len(self.class_names))
        loss_sum = 0

        for i in range(0, len(embeddings), self.batch_size):
            self.optimizer.zero_grad()
            indices = permutation[i : i + self.batch_size]
            batch_X = X[indices].float()
            batch_y = y_one_hot[indices]
            y_pred = self.model(batch_X)
            if self.loss_function == torch.nn.CrossEntropyLoss:
                loss = self.loss_function(y_pred, batch_y)
            else:
                loss = self.loss_function(
                    y_pred=y_pred,
                    y_true=batch_y,
                    labels=labels,
                    weights=[0.1, 0.1, 0.1, 0.7],
                    loss_function=torch.nn.CrossEntropyLoss,
                )
            loss_sum += loss
            loss.backward()
            self.optimizer.step()

        loss_avg = float(loss_sum / (math.ceil(len(embeddings) / self.batch_size)))
        model_specific_metrics = {"loss_avg": loss_avg}
        return model_specific_metrics

    def predict(self, embeddings: np.ndarray) -> Prediction:
        predicted_probabilities = self.model(torch.from_numpy(embeddings).float().to(self.device))
        df = pd.DataFrame(
            predicted_probabilities, columns=[str(label) for label in self.class_names]
        ).astype("float")
        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError


def hierarchical_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    labels: List[str],
    weights: np.ndarray,
    loss_function,
) -> torch.Tensor:
    loss = 0
    for i, level in enumerate(["C", "A", "T", "H"]):
        pred = _get_predictions_for_level(cath_level=level, y_pred=y_pred, labels=labels)
        true = _get_predictions_for_level(cath_level=level, y_pred=y_true, labels=labels)

        print(f"type(pred) = {type(pred)}")
        print(f"type(true) = {type(true)}")
        print(f"type(loss_function(pred, true)) = {type(loss_function(pred, true))}")
        print(f"loss_function(pred, true) = {loss_function(pred, true)}")

        loss += weights[i] * loss_function(pred, true)

    return torch.Tensor(loss)


def _get_predictions_for_level(
    cath_level: Literal["C", "A", "T", "H"], y_pred: torch.Tensor, labels: List[str]
) -> torch.Tensor:
    level = "CATH".index(cath_level)
    new_preds = []

    for proba_dist in y_pred:
        prev_label = ".".join(labels[0].split(".")[: level + 1])
        new_pred = []
        p_level = 0
        for i, p in enumerate(proba_dist):
            if labels[i].startswith(prev_label):
                p_level += int(p)
            else:
                new_pred.append(p_level)
                p_level = int(p)
                prev_label = ".".join(labels[i].split(".")[: level + 1])

        new_pred.append(p_level)
        new_preds.append(new_pred)

    return torch.tensor(new_preds)


class DistanceModel(ModelInterface):
    def __init__(
        self, embeddings: np.ndarray, labels: List[str], class_names: List[str], distance_ord: int
    ):
        self.device = torch_utils.get_device()
        self.X_train_tensor = torch.tensor(embeddings).to(self.device)
        self.y_train = labels
        self.class_names = sorted(list(set([str(cn) for cn in class_names])))
        if distance_ord < 0:
            raise ValueError(f"Distance order must be >= 0, but it is: {distance_ord}")
        self.distance_ord = distance_ord

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        return {}

    def predict(self, embeddings: np.ndarray) -> Prediction:
        emb_tensor = torch.tensor(embeddings).to(self.device)
        pdist = torch.nn.PairwiseDistance(p=self.distance_ord, eps=1e-08).to(self.device)

        distances = [
            [pdist(emb, emb_lookup) for emb_lookup in self.X_train_tensor] for emb in emb_tensor
        ]
        distances = np.array(torch.tensor(distances).cpu())

        pred_labels = np.array([self.y_train[i] for i in np.argmin(distances, axis=1)])
        pred_indices = [self.class_names.index(label) for label in pred_labels]
        pred = np.zeros(shape=(len(embeddings), len(self.class_names)))
        for row, index in enumerate(pred_indices):
            pred[row, index] = 1

        df = pd.DataFrame(data=pred, columns=self.class_names)
        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError
