from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import ray
from ray import tune
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB

from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
)
from gobi_cath_classification.pipeline.data_loading import (
    load_data,
    check_if_cath_level_is_valid,
    label_for_level,
)


class Prediction:
    def __init__(self, probabilities: pd.DataFrame):
        self.probabilities = probabilities
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


class ModelInterface:
    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> None:
        """
        Trains the model.

        Args:
            embeddings:
                2D array with shape (number of embeddings, 1024)
            labels:
                2D array with shape (number of embeddings, number of classes) of
                one-hot-encoded labels
            sample_weights:
                1D array with sample weights, shape (number of embeddings)

        """
        raise NotImplementedError

    def predict_proba(self, embeddings: np.ndarray) -> Prediction:
        """
        Predicts probabilities for the CATH superfamily labels.

        Args:
            embeddings:
                2D array with shape (number of embeddings, 1024)

        Returns:
            Pandas DataFrame with shape (number of embeddings, number of classes) of
            probabilities. Each column corresponds to one CATH superfamily.
        """
        raise NotImplementedError

    def save_checkpoint(self, save_to_dir: Path):
        """

        Save a checkpoint to given directory.

        """
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        """

        Load model from given checkpoint file(s),

        """
        raise NotImplementedError


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


def accuracy_for_level(y_true: List[str], y_pred: List[str], cath_level: str) -> float:
    check_if_cath_level_is_valid(cath_level=cath_level)
    if cath_level == "H":
        return accuracy_score(y_true=y_true, y_pred=y_pred)
    else:
        y_true_for_level = [label_for_level(label=label, cath_level=cath_level) for label in y_true]
        y_pred_for_level = [label_for_level(label=label, cath_level=cath_level) for label in y_pred]
        return accuracy_score(y_true=y_true_for_level, y_pred=y_pred_for_level)


def evaluate(y_true: List[str], y_pred: Prediction) -> Dict[str, float]:
    y_proba = y_pred.probabilities
    y_labels = y_pred.argmax_labels()
    eval_dict = {
        "accuracy_c": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="C"),
        "accuracy_a": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="A"),
        "accuracy_t": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="T"),
        "accuracy_h": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="H"),
        # "roc_ovr_micro_avg": roc_auc_score(y_true=y_true, y_score=y_proba, average="micro",
        #                                    multi_class="ovr"),
        # "roc_ovr_macro_avg": roc_auc_score(y_true=y_true, y_score=y_proba, average="macro",
        #                                    multi_class="ovr"),
    }
    eval_dict["accuracy_avg"] = (
        eval_dict["accuracy_c"]
        + eval_dict["accuracy_a"]
        + eval_dict["accuracy_t"]
        + eval_dict["accuracy_h"]
    ) / 4
    # TODO roc_50
    return eval_dict


def training_function(config: dict):
    # data
    # TODO implement function that loads a small sample for quick test

    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_set = load_data(data_dir=data_dir, without_duplicates=True)
    num_classes = len(data_set.all_labels_train_sorted)
    print(f"num_classes = {num_classes}")

    # Hyperparameters
    print(f"config = {config}")

    if config["class_weights"] == "none":
        sample_weights = None
    elif config["class_weights"] == "inverse":
        sample_weights = compute_inverse_sample_weights(labels=data_set.y_train)
    elif config["class_weights"] == "sqrt_inverse":
        sample_weights = np.sqrt(compute_inverse_sample_weights(labels=data_set.y_train))
    else:
        raise ValueError(f'Class weights do not exist: {config["class_weights"]}')

    num_epochs = config["model"]["num_epochs"]
    model_class = config["model"]["model_class"]

    if model_class == RandomForestModel.__name__:
        model = RandomForestModel(max_depth=config["model"]["max_depth"])
    elif model_class == "GaussianNaiveBayes":
        model = GaussianNaiveBayesModel()
    elif model_class == "NeuralNetwork":
        raise NotImplementedError
    else:
        raise ValueError(f"Model class {model_class} does not exist.")

    print(f"Training model {model.__class__.__name__}...")
    for epoch in range(num_epochs):
        n = 1000
        model.train_one_epoch(
            embeddings=data_set.X_train[:n, :],
            labels=data_set.y_train[:n],
            sample_weights=sample_weights[:n],
        )

        print(f"Predicting for X_val with model {model.__class__.__name__}...")
        y_pred_val = model.predict_proba(embeddings=data_set.X_val)

        # evaluate and save results in ray tune
        eval_dict = evaluate(y_true=data_set.y_val, y_pred=y_pred_val)

        tune.report(**eval_dict)


def main():
    # TODO ray tune checkpoint models ?
    ray.init()
    analysis = tune.run(
        training_function,
        num_samples=1,
        config={
            "class_weights": tune.choice(["none", "inverse", "sqrt_inverse"]),
            "model": tune.grid_search(
                [
                    {
                        "model_class": RandomForestModel.__name__,
                        "max_depth": tune.choice([1, 2, 3]),
                        "num_epochs": 1,
                    },
                    {"model_class": "GaussianNaiveBayes", "num_epochs": 1},
                    {
                        "model_class": "NeuralNetwork",
                        "lr": tune.choice([1e-4, 1e-5, 1e-6]),
                        "num_epochs": 100,
                    },
                ]
            ),
        },
    )

    print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df


if __name__ == "__main__":
    main()
