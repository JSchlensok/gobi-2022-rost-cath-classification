from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import torch
from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds
from gobi_cath_classification.pipeline.data import Dataset
from gobi_cath_classification.pipeline.sample_weights import compute_class_counts


class RandomBaseline(ModelInterface):
    """
    The RandomBaseline model does not need the train_one_epoch function
    it only predicts a random class for each input of the predict method
    to predict a random class, we implement two different methods:
        1. The class sizes are ignored and we just generate random numbers
        2. The class sizes are taken into account during the prediction generation
    We therefore have the parameter class_balance to differ between these two methods
    """

    def __init__(
        self,
        data: Dataset,
        class_balance: bool,
        rng: np.random.RandomState,
        random_seed: int = 42,
    ):
        """
        Args:
            data: a DataSplits object created from the data in the data folder
            class_balance: differentiate between the two methods mentioned above
            rng: random seed setting
            random_seed: random seed setting
        """
        self.data = data
        self.class_balance = class_balance
        self.device = torch_utils.get_device()
        self.random_seed = random_seed
        self.rng = rng
        print(f"rng = {rng}")
        set_random_seeds(seed=random_seed)

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        pass

    def predict(self, embeddings: np.ndarray) -> Prediction:
        """

        Args:
            embeddings: the input embeddings from which we want to generate a random prediction

        Returns: a Prediction object (panda df) containing for each column a predicted value and each row
        is a element from the input embeddings

        """

        class_names = self.data.train_labels

        # method without class_balance: every class has equal probability
        if not self.class_balance:
            # generate random probabilities for each class and each validation_label
            predicted_probabilities = np.random.uniform(
                0, 1, (embeddings.shape[0], len(class_names))
            )
            df = pd.DataFrame(
                data=predicted_probabilities, columns=[str(label) for label in class_names]
            )

        # method with class_balance: every class is weighted with the corresponding number of labels in
        # its class
        else:
            # count the labels in the training set and create a sorted list with the counts of the lables
            class_weights = compute_class_counts(self.data.y_train)
            predicted_probabilities = np.zeros((embeddings.shape[0], len(class_names)))
            # by creating a random number between 0 and the total number of labels in the train set, the probability of
            # hitting a certain label is "anzahl label" / "anzahl alle labels"

            for row in predicted_probabilities:
                rand = np.random.randint(low=0, high=np.sum(class_weights) + 1)
                counter = 0
                for i in range(len(row)):
                    counter += class_weights[i]
                    if rand < counter:
                        row[i] = 1
                        break
                    else:
                        counter += class_weights[i]

            """ MinMax Scaler did produce same results
            assert predicted_probabilities.shape[1] == class_weights.shape[0]
            # multiply each row of the prediction with the counts of the classes
            predicted_probabilities = np.multiply(predicted_probabilities, class_weights)

            # scale the data to a probability between 0 and 1 by dividing with the total length of
            min_max_scaler = preprocessing.MinMaxScaler()
            predicted_probabilities = min_max_scaler.fit_transform(predicted_probabilities)
            """

            # create the probability data frame
            df = pd.DataFrame(
                data=predicted_probabilities, columns=[str(label) for label in class_names]
            )

        return Prediction(probabilities=df)

    def save_checkpoint(self, save_to_dir: Path):
        raise NotImplementedError

    def load_model_from_checkpoint(self, load_from_dir: Path):
        raise NotImplementedError


class ZeroRate(ModelInterface):
    """
    The ZeroRate baseline predicts the largest class for each instance
    It counts the class labels and extracts it

    """

    def __init__(
        self,
        data: Dataset,
        rng: np.random.RandomState,
        random_seed: int = 42,
    ):
        """
        Args:
            data: a DataSplits object created from the data in the data folder
            rng: random seed setting
            random_seed: random seed setting
        """
        self.data = data
        self.device = torch_utils.get_device()
        self.random_seed = random_seed
        self.rng = rng
        print(f"rng = {rng}")
        set_random_seeds(seed=random_seed)

    def train_one_epoch(
        self,
        embeddings: np.ndarray,
        embeddings_tensor: torch.Tensor,
        labels: List[str],
        sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        pass

    def predict(self, embeddings: np.ndarray) -> Prediction:
        """
        The prediction method extracts the class label with the highest number of instances and predicts it for each of
        the inputs
        Args:
            embeddings: embedding vector of the validation set

        Returns:
            Prediction object with only the largest class predicted
        """

        class_names = self.data.train_labels
        class_weights = compute_class_counts(self.data.y_train)
        max_index = np.argmax(class_weights)
        predicted_probabilities = np.zeros((embeddings.shape[0], len(self.data.train_labels)))
        for row in predicted_probabilities:
            row[max_index] = 1

        df = pd.DataFrame(
            data=predicted_probabilities, columns=[str(label) for label in class_names]
        )
        return Prediction(probabilities=df)

    def load_model_from_checkpoint(self, load_from_dir: Path):
        pass

    def save_checkpoint(self, save_to_dir: Path):
        pass
