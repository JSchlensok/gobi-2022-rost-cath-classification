import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.pipeline.model_interface import Prediction

from .utils import CATHLabel

labels_train_H = sorted(
    [
        CATHLabel(label)
        for label in [
            "1.400.35.20",
            "5.20.20.400",
            "3.20.100.25",
            "5.20.30.300",
            "3.200.100.20",
            "2.20.300.25",
        ]
    ]
)

labels_train_T = sorted([label["T"] for label in labels_train_H])
labels_train_A = sorted([label["A"] for label in labels_train_H])
labels_train_C = sorted([label["C"] for label in labels_train_H])


def test_accuracy_for_level_H():
    print(labels_train_H)
    print(labels_train_T)
    print(labels_train_A)
    print(labels_train_C)

    # labels represented as numbers
    y_true = np.array([4, 3, 2, 5, 1, 2, 1, 3, 4, 0])
    y_pred_proba = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.8, 0.2],  # correct
            [0.0, 0.0, 0.0, 0.2, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.7, 0.0, 0.3],
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.8],  # correct
            [0.0, 0.7, 0.0, 0.3, 0.0, 0.0],  # correct
            [0.0, 0.0, 0.4, 0.5, 0.0, 0.1],
            [0.0, 0.3, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.6, 0.4, 0.0],
            [0.5, 0.4, 0.1, 0.0, 0.0, 0.0],
        ]
    )

    y_pred_argmax = np.argmax(y_pred_proba, axis=1)

    # labels represented as strings
    y_true_str = [
        "3.200.100.20",
        "5.20.30.300",
        "3.20.100.25",
        "2.20.300.25",
        "5.20.20.400",
        "3.20.100.25",
        "5.20.20.400",
        "5.20.30.300",
        "3.200.100.20",
        "1.400.35.20",
    ]

    y_pred_str = [
        "3.200.100.20",
        "2.20.300.25",
        "5.20.30.300",
        "2.20.300.25",
        "5.20.20.400",
        "5.20.30.300",
        "3.20.100.25",
        "3.20.100.25",
        "5.20.30.300",
        "5.20.30.300",
    ]
    acc_H = accuracy_score(y_true=y_true_str, y_pred=y_pred_str)
    assert acc_H == 0.3


def test_evaluate():
    columns = ["1.25.300.45", "2.25.400.10", "2.25.500.10", "3.10.25.300"]
    y_true = ["1.25.300.45", "2.25.400.10", "1.25.300.450"]
    prediction = Prediction(
        probabilities=pd.DataFrame(
            np.array(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.2, 0.2, 0.5, 0.1],
                    [0.05, 0.7, 0.05, 0.2],
                ]
            ),
            columns=columns,
        )
    )
    eval_dict = evaluate(
        y_true=[CATHLabel(label) for label in y_true],
        y_pred=prediction,
        class_names_training=[CATHLabel(label) for label in columns],
    )
    assert eval_dict["accuracy_h"] == 0.5
    assert eval_dict["accuracy_t"] == 1 / 3
    assert eval_dict["accuracy_a"] == 2 / 3
    assert eval_dict["accuracy_c"] == 2 / 3
