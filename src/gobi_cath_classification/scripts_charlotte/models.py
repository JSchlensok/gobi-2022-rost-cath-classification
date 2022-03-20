import math
import os

from pathlib import Path
from typing import List, Optional, Dict, Callable
from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from torch import nn
from torch.nn.functional import one_hot

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline.utils import torch_utils, CATHLabel
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds


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
        print(f"Attempting to save model 'RandomForestModel' in file model_object.model")
        print(f"Saving into directory: '{save_to_dir}'")
        checkpoint_file_path = os.path.join(save_to_dir, "model_object.model")
        try:
            torch.save(self, checkpoint_file_path)
            print(f"Checkpoint saved to: {checkpoint_file_path}")
        except:
            print(f"Failed to save model 'RandomForestModel'")

    def load_model_from_checkpoint(self, checkpoint_dir: Path):
        print(f"Attempting to reload model 'RandomForestModel' from file: {checkpoint_dir}")
        try:
            model_file_path = os.path.join(checkpoint_dir, "model_object.model")
            model = torch.load(model_file_path)
            print(f"Successfully read in model!")
            return model
        except:
            print(f"Failed to read in model!")
            return None


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
        print(f"Attempting to save model 'GaussianNaiveBayesModel' in file model_object.model")
        print(f"Saving into directory: '{save_to_dir}'")
        checkpoint_file_path = os.path.join(save_to_dir, "model_object.model")
        try:
            torch.save(self, checkpoint_file_path)
            print(f"Checkpoint saved to: {checkpoint_file_path}")
        except:
            print(f"Failed to save model 'GaussianNaiveBayesModel'")

    def load_model_from_checkpoint(self, checkpoint_dir: Path):
        print(f"Attempting to reload model 'GaussianNaiveBayesModel' from file: {checkpoint_dir}")
        try:
            model_file_path = os.path.join(checkpoint_dir, "model_object.model")
            model = torch.load(model_file_path)
            print(f"Successfully read in model!")
            return model
        except:
            print(f"Failed to read in model!")
            return None


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
        print(f"Attempting to save model 'DistanceModel' in file model_object.model")
        print(f"Saving into directory: '{save_to_dir}'")
        checkpoint_file_path = os.path.join(save_to_dir, "model_object.model")
        try:
            torch.save(self, checkpoint_file_path)
            print(f"Checkpoint saved to: {checkpoint_file_path}")
        except:
            print(f"Failed to save model 'DistanceModel'")

    def load_model_from_checkpoint(self, checkpoint_dir: Path):
        print(f"Attempting to reload model 'DistanceModel' from file: {checkpoint_dir}")
        try:
            model_file_path = os.path.join(checkpoint_dir, "model_object.model")
            model = torch.load(model_file_path)
            print(f"Successfully read in model!")
            return model
        except:
            print(f"Failed to read in model!")
            return None


class NeuralNetworkModel(ModelInterface):
    def __init__(
        self,
        lr: float,
        class_names: List[str],
        layer_sizes: List[int],
        dropout_sizes: List[Optional[float]],
        batch_size: int,
        optimizer: str,
        loss_function: Literal["CrossEntropyLoss", "HierarchicalLogLoss", "HierarchicalMSELoss"],
        class_weights: torch.Tensor,
        rng: np.random.RandomState,
        random_seed: int = 42,
        weight_decay: float = 0.0,
        loss_weights: torch.Tensor = None,
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
            nn.Linear(in_features=layer_sizes[-1], out_features=len(self.class_names)),
        )
        if dropout_sizes[-1] is not None:
            model.add_module(
                f"Dropout_{len(dropout_sizes) - 1}",
                nn.Dropout(
                    p=dropout_sizes[-1],
                ),
            )
        if loss_function is not "CrossEntropyLoss":
            model.add_module("Softmax", nn.Softmax())

        self.model = model.to(self.device)

        if loss_function == "CrossEntropyLoss":
            self.loss_function = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None,
            )
        elif loss_function == "HierarchicalLogLoss" or loss_function == "HierarchicalMSELoss":
            self.loss_function = HierarchicalLoss(
                loss_function=log_loss
                if loss_function == "HierarchicalLogLoss"
                else mean_squared_error,
                class_names=self.class_names,
                class_weights=class_weights.to(self.device) if class_weights is not None else None,
                hierarchical_weights=loss_weights.to(self.device),
                device=self.device,
            )
        else:
            raise ValueError(f"Loss_function is not valid: {loss_function}")

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
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

            loss = self.loss_function(y_pred, batch_y)

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

    def save_checkpoint(self, save_to_dir: Path) -> None:
        print(f"Attempting to save model 'NeuralNetworkModel' in file model_object.model")
        print(f"Saving into directory: '{save_to_dir}'")
        checkpoint_file_path = os.path.join(save_to_dir, "model_object.model")
        try:
            torch.save(self, checkpoint_file_path)
            print(f"Checkpoint saved to: {checkpoint_file_path}")
        except:
            print(f"Failed to save model 'NeuralNetworkModel'")

    def load_model_from_checkpoint(self, checkpoint_dir: Path):
        print(f"Attempting to reload model 'NeuralNetworkModel' from file: {checkpoint_dir}")
        try:
            model_file_path = os.path.join(checkpoint_dir, "model_object.model")
            model = torch.load(model_file_path)
            print(f"Successfully read in model!")
            return model
        except:
            print(f"Failed to read in model!")
            return None


def log_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor = None
) -> torch.Tensor:

    x_log_y = torch.special.xlogy(input=y_true, other=y_pred)

    if sample_weights is not None:
        sw_broadcasted = torch.broadcast_to(
            torch.reshape(sample_weights, (len(sample_weights), 1)), x_log_y.size()
        )
        x_log_y = torch.mul(sw_broadcasted, x_log_y)

    log_loss = (-1) * torch.mean(torch.sum(x_log_y, dim=1))

    return log_loss


def mean_squared_error(
    y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights=None
) -> torch.Tensor:
    return (y_pred - y_true).pow(2).sum()


class HierarchicalLoss:
    """

    Computes the weighted averaged accuracy over all levels.
    The goal is to 'punish' a model less for a prediction that's incorrect on the H-level, but
    correct on all other levels in comparison to a prediction which is incorrect on more levels
    than just the H-level.

    Example:
        y_true = "1.2.3.4"
        pred_1 = "1.2.3.999"
        pred_2 = "5.6.7.8"

        In this case we want to acknowledge that pred_1 is a better prediction than pred_2.
        Therefore the punishment for pred_2 should be bigger than for pred_1.

    """

    def __init__(
        self,
        loss_function: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        class_weights: torch.Tensor,
        hierarchical_weights: torch.Tensor,
        class_names: List[str],
        device,
    ):
        assert len(hierarchical_weights) == 4
        assert torch.allclose(
            torch.sum(hierarchical_weights).to(device), torch.tensor([1.0]).to(device)
        )
        self.loss_function = loss_function
        self.class_weights = class_weights.to(device)
        self.hierarchical_weights = hierarchical_weights.to(device)
        self.class_names = class_names
        self.device = device

        self.H_to_C_matrix = H_to_level_matrix(class_names=class_names, level="C").to(device)
        self.H_to_A_matrix = H_to_level_matrix(class_names=class_names, level="A").to(device)
        self.H_to_T_matrix = H_to_level_matrix(class_names=class_names, level="T").to(device)
        self.H_to_H_matrix = torch.eye(n=len(self.class_names)).to(self.device)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        matrices = [
            self.H_to_C_matrix,
            self.H_to_A_matrix,
            self.H_to_T_matrix,
            self.H_to_H_matrix,
        ]
        if self.class_weights is not None:
            sample_weights = torch.tensor(
                [self.class_weights[(y == 1.0).nonzero().item()] for y in y_true]
            ).to(self.device)
        else:
            sample_weights = None

        loss = torch.tensor([0]).float().to(self.device)
        for i, H_to_level_matrix in enumerate(matrices):
            loss += self.hierarchical_weights[i] * self.loss_function(
                torch.matmul(y_pred, H_to_level_matrix),
                torch.matmul(y_true, H_to_level_matrix),
                sample_weights,
            )
        return loss


def H_to_level_matrix(class_names: List[str], level: str) -> torch.Tensor:
    """

    Returns a matrix, wherein each row corresponds to a label on H-level, and each column
    corresponds to a label on C-, A- or T-level. The numbers in the matrix are either 0 or 1:
        - matrix(row, col) = 0 if label_in_col == label_for_level(label_in_row)
        - matrix(row, col) = 1 if label_in_col != label_for_level(label_in_row)
    The sum of each row be equal to 1, since each label on H-level can only belong to one label on
    C-, A- or T-level (see example below: if "1.2.2.4" belongs to "1" it can't belong to any of the
    other labels on C-level.)

    Example:

        rows := labels on H-level = ["1.2.2.4", "2.3.4.5", "2.3.9.9"]
        columns := labels on C-level = ["1", "3"]

                    "1"    "3"
        "1.2.2.4"    1      0       -> 1 + 0 = 1
        "3.4.4.5"    0      1       -> 0 + 1 = 1
        "3.4.9.9"    0      1       -> 0 + 1 = 1

    """
    assert level in ["C", "A", "T", "H"]
    class_names_level = sorted(list(set([str(CATHLabel(cn)[:level]) for cn in class_names])))
    matrix = []
    for cn in class_names:
        row = []
        for cn_level in class_names_level:
            row.append(1 if cn.startswith(cn_level) else 0)
        matrix.append(row)

    return torch.Tensor(matrix)


def compute_predictions_for_ensemble_model(
    predictions_from_models: List[Prediction], weights: np.ndarray
) -> Prediction:
    """

    Computes elementwise weighted average of the given probabilities and returns a new prediction
    with those probabilities.

    """
    np.testing.assert_allclose(
        actual=np.sum(weights), desired=1.0
    ), f"The given weights don't sum up to one, but instead to: {np.sum(weights)}"
    assert len(predictions_from_models) == len(weights), (
        f"The amount of given predictions does not equal the amount of given weights: "
        f"{len(predictions_from_models)} != {len(weights)}"
    )

    num_samples, num_labels = predictions_from_models[0].probabilities.shape
    col_names = predictions_from_models[0].probabilities.columns.tolist()
    ensemble_pred = []

    for i in range(num_samples):
        ensemble_pred_sample = np.zeros(shape=num_labels)
        for j, p in enumerate(predictions_from_models):
            weighted_pred = weights[j] * p.probabilities.iloc[[i]].to_numpy()
            ensemble_pred_sample = np.sum((ensemble_pred_sample, weighted_pred), axis=0).flatten()
        ensemble_pred.append(ensemble_pred_sample)

    ensemble_prediction = Prediction(
        probabilities=pd.DataFrame(data=np.array(ensemble_pred), columns=col_names)
    )
    return ensemble_prediction
