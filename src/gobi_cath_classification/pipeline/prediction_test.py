import os

from pathlib import Path

import numpy as np
import pandas as pd

from gobi_cath_classification.pipeline.prediction import (
    save_predictions,
    Prediction,
    read_in_predictions,
)


def test_save_and_read_in_prediction():
    filepath = Path(__file__).parent.parent.parent.parent
    filename = "pred_test.csv"
    pred_1 = Prediction(
        pd.DataFrame(
            data=np.array(
                [
                    [0.1, 0.1, 0.1, 0.7],
                    [0.05, 0.3, 0.6, 0.05],
                    [0.9, 0.01, 0.01, 0.08],
                ]
            ),
            columns=["1.20.35.10", "2.40.50.10", "3.20.25.400", "3.300.20.5"],
        )
    )
    save_predictions(pred=pred_1, directory=filepath, filename=filename)
    pred_2 = read_in_predictions(filepath=filepath / "pred_test.csv")
    os.remove(path=filepath / filename)

    pd.testing.assert_frame_equal(pred_1.probabilities, pred_2.probabilities)
