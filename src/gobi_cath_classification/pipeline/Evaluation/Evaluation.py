from typing import List
from typing_extensions import Literal
import numpy as np

from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from gobi_cath_classification.pipeline.utils import CATHLabel
from gobi_cath_classification.pipeline.model_interface import Prediction


class Evaluation:
    """
    The Evaluation class:
    - use this class to compute metrics for your model
    - supported metrics are: Accuracy, Matthews correlation coefficient, F1-score and Cohen's kappa score

    -> in the __init__ pass the given parameters
    -> with compute_metrics() you can specify the metrics you want to compute
    -> the metrics are then saved in the eval_dict of the Evaluation object in this structure
        eval_dict = {"accuracy": {"accuracy_C": acc_C, "accuracy_A": acc_A, ... , "accuracy_avg": acc_avg"},
                    "mcc": {"mcc_C": mcc_C, "mcc_A": mcc_A, ... , "mcc_avg": mcc_avg"},
                    "f1": {"f1_C": f1_C, "f1_A": acc_A, ... , "f1_avg": f1_avg"},
                    "kappa": {kappa_C": kappa_C, "kappa_A": kappa_A, ... , "kappa_avg": kappa_avg"}

    Note: only the metrics that were specified in the compute_metric() function are available in the dict
    """

    def __init__(
        self, y_true: List[CATHLabel], predictions: Prediction, train_labels: List[CATHLabel]
    ):
        """
        Create a Evaluation class with the true labels, the Prediction object from the predict method of the model
        and all the labels that wre used in the training

        Args:
            y_true: True labels
            predictions: Prediction object returned from the predict() method of the model
            train_labels: all labels that were used in the training
        """

        self.y_true = y_true
        self.yhat_probabilities = predictions.probabilities
        self.yhats = [CATHLabel(label) for label in predictions.argmax_labels()]
        self.train_labels = train_labels
        self.eval_dict = None
        self.error_dict = None

    def compute_metrics(
        self,
        accuracy: bool = False,
        mcc: bool = False,
        f1: bool = False,
        kappa: bool = False,
        _y_true: List = None,
        _y_hats: List = None,
    ):
        """
        Usage: set the desired metrics to "TRUE"
        -> creates an evaluation dictionary containing all the specified metrics

        Supported Metrics
        Args:
            accuracy: accuracy for each level in CATH
            mcc: matthews correlation coefficient for each level in CATH
            f1: the f1 score for each level in CATH
            kappa: cohen's kappa for each level in CATH
            _y_true: default=None (ignore: only for internal usage)
            _y_hats: default=None (ignore: only for internal usage)

        Returns:
            updates the evaluation dict with the chosen metrics

        """

        CATH_levels = ["C", "A", "T", "H"]
        assert (accuracy or mcc or f1 or kappa) is True, print(
            "At least on metric must be selected"
        )

        # set class parameters if no external lists are given
        bootstrap = True
        if _y_true is None and _y_hats is None:
            _y_true = self.y_true
            _y_hats = self.yhats
            bootstrap = False

        eval_dict = dict()
        bootstrap_dict = dict()

        # calculate the accuracy for each level
        if accuracy:
            # calculating accuracy using dict comprehension
            accuracy_dict = {
                f"accuracy_{i.lower()}": metric_for_level(
                    _y_true, _y_hats, self.train_labels, i, "acc"
                )
                for i in CATH_levels
            }

            # average accuracy over all levels
            accuracy_dict["accuracy_avg"] = sum(accuracy_dict.values()) / len(CATH_levels)

            eval_dict["accuracy"] = accuracy_dict
            bootstrap_dict["accuracy"] = accuracy_dict

        # calculate f1 for each level
        if mcc:
            mcc_dict = {
                f"mcc_{i.lower()}": metric_for_level(_y_true, _y_hats, self.train_labels, i, "mcc")
                for i in CATH_levels
            }
            mcc_dict["mcc_avg"] = sum(mcc_dict.values()) / len(CATH_levels)
            eval_dict["mcc"] = mcc_dict
            bootstrap_dict["mcc"] = mcc_dict

        # calculating f1 for each level
        if f1:
            f1_dict = {
                f"f1_{i.lower()}": metric_for_level(_y_true, _y_hats, self.train_labels, i, "f1")
                for i in CATH_levels
            }
            f1_dict["f1_avg"] = sum(f1_dict.values()) / len(CATH_levels)
            eval_dict["f1"] = f1_dict
            bootstrap_dict["f1"] = f1_dict

        # calculating cohen's kappa for each level
        if kappa:
            kappa_dict = {
                f"kappa_{i.lower()}": metric_for_level(
                    _y_true, _y_hats, self.train_labels, i, "kappa"
                )
                for i in CATH_levels
            }
            kappa_dict["kappa_avg"] = sum(kappa_dict.values()) / len(CATH_levels)
            eval_dict["kappa"] = kappa_dict
            bootstrap_dict["kappa"] = kappa_dict

        # if the bootstrap dict is required, return it
        if bootstrap:
            return bootstrap_dict
        else:
            self.eval_dict = eval_dict

    def compute_std_err(self, bootstrap_n: int = 1000):
        """
        compute the standard error for the metrics currently in the evaluation dict using 1000 bootstrap intervals
        in each bootstrap interval, choose samples with replacement and compute the given metric

        Args:
            bootstrap_n: number of bootstrap intervals

        Returns:
            creates a new dict with the 95% confidence intervals for the available metrics
        """

        # what metrics are available and what metrics are there in general
        available_metrics = [
            "accuracy" in self.eval_dict,
            "mcc" in self.eval_dict,
            "f1" in self.eval_dict,
            "kappa" in self.eval_dict,
        ]
        metrics = ["accuracy", "mcc", "f1", "kappa"]

        n_pred = len(self.y_true)
        indexes = range(n_pred)
        bootstrap_dicts = list()

        # only if the eval_dict contains metrics we can calculate the error
        if self.eval_dict is not None:
            for _ in range(bootstrap_n):
                # choose n_pred indices with replacement
                sample = np.random.choice(indexes, n_pred, replace=True)
                y_true = [self.y_true[idx] for idx in sample]
                y_hats = [self.yhats[idx] for idx in sample]
                tmp_eval = self.compute_metrics(*available_metrics, _y_true=y_true, _y_hats=y_hats)

                bootstrap_dicts.append(tmp_eval)

            # extract the bootstrapped metrics
            error_dict = {}

            for i, available in enumerate(available_metrics):
                if available:
                    error_dict[metrics[i]] = {}
                    for level in ["c", "a", "t", "h", "avg"]:

                        # for all available metrics calculate the standard error for all levels
                        error_dict[f"{metrics[i]}"][f"{metrics[i]}_{level}"] = 1.96 * np.std(
                            np.array(
                                [
                                    boot[metrics[i]][f"{metrics[i]}_{level}"]
                                    for boot in bootstrap_dicts
                                ]
                            ),
                            ddof=1,
                        )

            self.error_dict = error_dict
        else:
            raise ValueError("The eval_dict does not contain any metrics yet")


def metric_for_level(
    y_true: List[CATHLabel],
    y_hat: List[CATHLabel],
    train_labels: List[CATHLabel],
    cath_level: Literal["C", "A", "T", "H"],
    metric: Literal["acc", "mcc", "f1", "kappa"],
) -> float:
    """
    Calculates the specific metric according to a given CATH-level.

    Args:
        y_true:
            List of labels (List[CATHLabel]): ground truth
        y_hat:
            List of labels (List[str]): prediction
        train_labels:
            Alphabetically sorted list of all labels that occurred in training
        cath_level:
            Literal["C", "A", "T", "H"]
        metric:
            the chosen metric is calculated for the specified level in the CATH hierarchy

    Returns:
        chosen metric score for each level in CATH and the mean

    """

    class_names_for_level = list(set([label[cath_level] for label in train_labels]))
    y_true_for_level = [str(label[cath_level]) for label in y_true]
    y_pred_for_level = [str(label[cath_level]) for label in y_hat]

    # delete all entries where the ground truth label does not occur in training class names.
    n = len(y_true_for_level) - 1
    for i in range(n, -1, -1):
        if y_true_for_level[i] not in class_names_for_level:
            del y_true_for_level[i]
            del y_pred_for_level[i]

    assert len(y_true_for_level) == len(y_pred_for_level)

    # compute the specified metric
    if metric == "acc":
        return accuracy_score(
            y_true=y_true_for_level,
            y_pred=y_pred_for_level,
        )
    if metric == "mcc":
        return matthews_corrcoef(y_true=y_true_for_level, y_pred=y_pred_for_level)
    if metric == "f1":
        return f1_score(
            y_true=y_true_for_level,
            y_pred=y_pred_for_level,
            # TODO: changing averaging technique
            average="macro",
        )
    if metric == "kappa":
        return cohen_kappa_score(y1=y_true_for_level, y2=y_pred_for_level)
