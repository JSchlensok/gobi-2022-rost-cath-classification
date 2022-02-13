import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from gobi_cath_classification.pipeline.data_loading import label_for_level
from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.pipeline.model_interface import Prediction

labels_train_H = sorted(
    list(
        set(
            [
                "1.400.35.20",
                "5.20.20.400",
                "3.20.100.25",
                "5.20.30.300",
                "3.200.100.20",
                "2.20.300.25",
            ]
        )
    )
)

labels_train_T = sorted(
    list(set([label_for_level(label, cath_level="T") for label in labels_train_H]))
)

labels_train_A = sorted(
    list(set([label_for_level(label, cath_level="A") for label in labels_train_H]))
)

labels_train_C = sorted(
    list(set([label_for_level(label, cath_level="C") for label in labels_train_H]))
)


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
    y_true = ["1.25.300.45", "2.25.400.10", "1.25.300.45"]
    prediction = Prediction(
        probabilities=pd.DataFrame(
            np.array(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.2, 0.5, 0.2, 0.1],
                    [0.05, 0.7, 0.05, 0.2],
                ]
            ),
            columns=columns,
        )
    )
    print(evaluate(y_true=y_true, y_pred=prediction))