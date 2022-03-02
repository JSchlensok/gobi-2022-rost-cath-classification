from typing import List, Optional
from typing_extensions import Literal

from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from gobi_cath_classification.pipeline.utils import CATHLabel
from gobi_cath_classification.pipeline.model_interface import Prediction


class Evaluation:
    def __init__(self,
                 y_true: List[CATHLabel],
                 predictions: Prediction,
                 train_labels: List[CATHLabel]):

        self.y_true = y_true
        self.yhat_probabilities = predictions.probabilities
        self.yhats = [CATHLabel(label) for label in predictions.argmax_labels()]
        self.train_labels = train_labels
        self.eval_dict = None

    def compute_metrics(self,
                        accuracy: bool = False,
                        mcc: bool = False,
                        f1: bool = False,
                        kappa: bool = False
                        ) -> None:
        """
        Usage: set the desired metrics to "TRUE"
        -> creates an evaluation dictionary containing all the specified metrics

        Supported Metrics
        Args:
            accuracy: accuracy for each level in CATH
            mcc: matthews correlation coefficient for each level in CATH
            f1: the f1 score for each level in CATH
            kappa: cohen's kappa for each level in CATH

        Returns:
            updates the evaluation dict with the chosen metrics

        """
        assert (accuracy or mcc or f1) is True, print("At least on metric must be selected")

        self.eval_dict = {}
        CATH_levels = ["C", "A", "T", "H"]

        # calculate the accuracy for each level
        if accuracy:
            # calculating accuracy using dict comprehension
            accuracy_dict = {
                f"accuracy_{i.lower()}":
                    metric_for_level(self.y_true, self.yhats, self.train_labels, i, "acc")
                for i in CATH_levels}

            # average accuracy over all levels
            accuracy_dict["accuracy_avg"] = sum(accuracy_dict.values()) / len(CATH_levels)
            self.eval_dict["accuracy"] = accuracy_dict

        # calculate f1 for each level
        if mcc:
            mcc_dict = {
                f"mcc_{i.lower()}":
                    metric_for_level(self.y_true, self.yhats, self.train_labels, i, "mcc")
                for i in CATH_levels
            }
            mcc_dict["mcc_avg"] = sum(mcc_dict.values()) / len(CATH_levels)
            self.eval_dict["mcc"] = mcc_dict

        # calculating f1 for each level
        if f1:
            f1_dict = {
                f"f1_{i.lower()}":
                    metric_for_level(self.y_true, self.yhats, self.train_labels, i, "f1")
                for i in CATH_levels
            }
            f1_dict["f1_avg"] = sum(f1_dict.values()) / len(CATH_levels)
            self.eval_dict["f1"] = f1_dict

        # calculating cohen's kappa for each level
        if kappa:
            kappa_dict = {
                f"kappa_{i.lower()}":
                    metric_for_level(self.y_true, self.yhats, self.train_labels, i, "kappa")
                for i in CATH_levels
            }
            kappa_dict["kappa_avg"] = sum(kappa_dict.values()) / len(CATH_levels)
            self.eval_dict["kappa"] = kappa_dict


def metric_for_level(
    y_true: List[CATHLabel],
    y_hat: List[CATHLabel],
    train_labels: List[CATHLabel],
    cath_level: Literal["C", "A", "T", "H"],
    metric: Literal["acc", "mcc", "f1", "kappa"]
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
    y_true_for_level = [label[cath_level] for label in y_true]
    y_pred_for_level = [label[cath_level] for label in y_hat]

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
            y_true=[str(label) for label in y_true_for_level],
            y_pred=[str(label) for label in y_pred_for_level],
        )
    if metric == "mcc":
        return matthews_corrcoef(
            y_true=[str(label) for label in y_true_for_level],
            y_pred=[str(label) for label in y_pred_for_level]
        )
    if metric == "f1":
        return f1_score(
            y_true=[str(label) for label in y_true_for_level],
            y_pred=[str(label) for label in y_pred_for_level],
            # TODO: changing averaging technique
            average="macro"
        )
    if metric == "kappa":
        return cohen_kappa_score(
            y1=[str(label) for label in y_true_for_level],
            y2=[str(label) for label in y_pred_for_level]
        )
