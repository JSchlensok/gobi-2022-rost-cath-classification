from typing import List, Dict

from sklearn.metrics import accuracy_score
from .utils import CATHLabel

from gobi_cath_classification.pipeline.model_interface import Prediction


def accuracy_for_level(
    y_true: List[str], y_pred: List[str], class_names_training: List[str], cath_level: str
) -> float:
    """

    Calculates accuracy according to a given CATH level.

    Args:
        y_true:
            List of labels (List[str]): ground truth
        y_pred:
            List of labels (List[str]): prediction
        class_names_training:
            Alphabetically sorted list of all labels that occured in training
        cath_level:
            0 for 'C' in 'CATH'
            1 for 'A' in 'CATH'
            2 for 'T' in 'CATH'
            3 for 'H' in 'CATH'

    Returns:
        Accuracy score

    """

    class_names_for_level = list(
        set([CATHLabel(label)[cath_level] for label in class_names_training])
    )
    y_true_for_level = [CATHLabel(label)[cath_level] for label in y_true]
    y_pred_for_level = [CATHLabel(label)[cath_level] for label in y_pred]

    # delete all entries where the ground truth label does not occur in training class names.
    n = len(y_true_for_level) - 1
    for i in range(n, -1, -1):
        if y_true_for_level[i] not in class_names_for_level:
            del y_true_for_level[i]
            del y_pred_for_level[i]

    assert len(y_true_for_level) == len(y_pred_for_level)

    return accuracy_score(
        y_true=[str(label) for label in y_true_for_level],
        y_pred=[str(label) for label in y_pred_for_level],
    )


def evaluate(
    y_true: List[str], y_pred: Prediction, class_names_training: List[str]
) -> Dict[str, float]:
    y_proba = y_pred.probabilities
    y_labels = y_pred.argmax_labels()
    eval_dict = {
        "accuracy_c": accuracy_for_level(
            y_true=y_true,
            y_pred=y_labels,
            class_names_training=class_names_training,
            cath_level="C",
        ),
        "accuracy_a": accuracy_for_level(
            y_true=y_true,
            y_pred=y_labels,
            class_names_training=class_names_training,
            cath_level="A",
        ),
        "accuracy_t": accuracy_for_level(
            y_true=y_true,
            y_pred=y_labels,
            class_names_training=class_names_training,
            cath_level="T",
        ),
        "accuracy_h": accuracy_for_level(
            y_true=y_true,
            y_pred=y_labels,
            class_names_training=class_names_training,
            cath_level="H",
        ),
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
    return eval_dict
