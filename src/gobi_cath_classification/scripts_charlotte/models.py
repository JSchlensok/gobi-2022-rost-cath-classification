from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction


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
