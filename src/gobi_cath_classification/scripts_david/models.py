# Import basic functionalities
from pathlib import Path
from typing import List, Optional

# Import classes specifically needed for machine learning
import numpy as np
import pandas as pd
import torch

# Import own classes needed in this script
from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import set_random_seeds

# Import libraries for machine learning from scikit learn and torch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from torch import nn
from torch.nn.functional import one_hot

class SupportVectorMachine(ModelInterface):
    ########################################################################################
    # CLASS NAME        : SupportVectorMachine
    # IMPLEMENTS        : ModelInterface
    # DESCRIPTION       : Class to implement a Support Vector machine as machine learning
    #                     model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 18.02.2022
    # UPDATE            : ---
    ########################################################################################

    def __init__(self, c=1.0, kernel="rbf", degree=3, gamma="scale"):
        # ####################################################################################### FUNCTION NAME     :
        # __init__() INPUT PARAMETERS  : self OUTPUT PARAMETERS : none DESCRIPTION       : Constructor to create
        # instance of class SupportVectorMachine AUTHOR            : D. Mauder CREATE DATE       : 18.02.2022 UPDATE
        # : 20.02.2022 - probability = True eingefügt damit predict_proba ausgeführt werden kann : 21.02.2022 Model
        # Cache auf 1000 MB erweitert
        # ####################################################################################### Parameter
        # description for SVMs Kernel: The main  function of  the  kernel is to transform the given dataset input
        # data into the required form. There are various types of functions such as linear, polynomial, and radial
        # basis function(RBF).Polynomial and RBF are useful for non - linear hyperplane.Polynomial and RBF kernels
        # compute the separation line in the higher dimension.In some of the applications, it is suggested to use a
        # more complex kernel to separate the classes that are curved or nonlinear.This transformation can lead to
        # more accurate classifiers.
        #
        # Regularization: Regularization parameter in python's Scikit-learn C parameter used to maintain
        # regularization. Here C is the penalty parameter, which represents mis classification or error term. The mis
        # classification or error term tells the SVM optimization how much error is bearable. This is how you can
        # control the trade-off between decision boundary and mis classification term. A smaller value of C creates a
        # small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
        #
        # Gamma: A lower value  of Gamma will loosely fit the training dataset, whereas a higher value of gamma will
        # exactly fit the training dataset, which causes over - fitting.In other words, you can say a low value of gamma
        # considers only nearby  points in calculating the separation line, while the a value of gamma considers all the
        # data points in the calculation of the separation line.

        self.model = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, probability=True)
        self.model.cache_size = 1000  # Make more MBs of RAM available for model cache

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> None:
        ########################################################################################
        # FUNCTION NAME     : train_one_epoch()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Train one single epoch in SVM-model
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################

        self.model.fit(X=embeddings[:1000], y=labels[:1000])

    def predict(self, embeddings: np.ndarray) -> Prediction:
        ########################################################################################
        # FUNCTION NAME     : predict()
        # INPUT PARAMETERS  : self
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Create predictions with SVM-model given correct input parameters
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : 20.02.2022 - predict durch predict_proba ersetzt
        ########################################################################################

        predictions = self.model.predict_proba(X=embeddings)
        df = pd.DataFrame(data=predictions, columns=self.model.classes_)
        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        ########################################################################################
        # FUNCTION NAME     : save_checkpoint()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Save the current state of the model to prevent information loss
        #                     in case of disturbances in program flow
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################

        # Checkpoint-Funktion ist für eine SVM nicht umsetzbar,
        # da nicht in Epochen trainiert wird sondern in nur einem Schritt

        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        ########################################################################################
        # FUNCTION NAME     : load_model_from_checkpoint()
        # INPUT PARAMETERS  : none
        # OUTPUT PARAMETERS : none
        # DESCRIPTION       : Load a specific, previously saved, state of the model
        # AUTHOR            : D. Mauder
        # CREATE DATE       : 18.02.2022
        # UPDATE            : ---
        ########################################################################################

        # Checkpoint-Funktion ist für eine SVM nicht umsetzbar,
        # da nicht in Epochen trainiert wird sondern in nur einem Schritt

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
                embeddings_tensor: torch.Tensor,
                labels: List[str],
                sample_weights: Optional[np.ndarray],
        ) -> None:
            self.model.fit(X=embeddings, y=labels, sample_weight=sample_weights)

        def predict(self, embeddings: np.ndarray) -> Prediction:
            predictions = self.model.predict(X=embeddings)
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

        def predict(self, embeddings: np.ndarray) -> Prediction:
            predictions = self.model.predict(X=embeddings)
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
                class_weights: torch.Tensor,
                rng: np.random.RandomState,
                random_seed: int = 42,
        ):
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
                model.add_module(f"ReLU_{i}", nn.ReLU())

            model.add_module(
                f"Linear_{len(layer_sizes) - 1}",
                nn.Linear(in_features=layer_sizes[-1], out_features=len(self.class_names)).to(
                    self.device
                ),
            )

            model.add_module("Softmax", nn.Softmax())
            self.model = model.to(self.device)
            self.loss_function = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None,
            )
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
                indices = permutation[i: i + self.batch_size]
                batch_X = X[indices].float()
                batch_y = y_one_hot[indices]
                y_pred = self.model(batch_X)
                loss = self.loss_function(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()

        def predict(self, embeddings: np.ndarray) -> Prediction:
            predicted_probabilities = self.model(torch.from_numpy(embeddings).float().to(self.device))
            df = pd.DataFrame(predicted_probabilities, columns=self.class_names).astype("float")
            return Prediction(probabilities=df)

        def save_checkpoint(self, save_to_dir: Path):
            raise NotImplementedError

        def load_model_from_checkpoint(self, load_from_dir: Path):
            raise NotImplementedError
