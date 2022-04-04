import math
import pickle
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_inverse_class_weights,
)

from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.rnn.models import RNNModel, BRNN, BRNN_embedded, RNN_embedded
from gobi_cath_classification.rnn.pipeline import load_data
from gobi_cath_classification.pipeline.Evaluation import Evaluation
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds
from gobi_cath_classification.rnn.models import RNNModel, BRNN, one_hot_encode
from gobi_cath_classification.rnn.pipeline import load_data
from gobi_cath_classification.pipeline.data.data_loading import DATA_DIR
from gobi_cath_classification.pipeline.data.Dataset import Dataset
from gobi_cath_classification.pipeline import prediction

_, _, train_labels, _, _, X_test, y_test = load_data(
    DATA_DIR,
    np.random.RandomState(42),
    without_duplicates=True,
    load_strings=False,
    reloading_allowed=True,
)
class_names = sorted(set(train_labels))

models = []
args = sys.argv
if len(args) > 2 and (args[1] == "-m" or args[1] == "--model"):
    model = torch.load(args[2])
    files = [Path(args[2])]
    models.append(model)
else:
    p = (DATA_DIR / "models").glob("**/*")
    files = [x for x in p if x.is_file()]
    models = [torch.load(f) for f in files]

for i in range(len(models)):
    model = models[i]
    model_name = files[i].name
    print(f"Predicting for model{model_name}")
    y_pred = model.predict(X_test)

    evaluation = Evaluation(
        y_true=y_test, predictions=y_pred, train_labels=class_names, model_name=model_name
    )
    print("Computing scores...")
    evaluation.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
    # print("Computing error...")
    # evaluation.compute_std_err()

    print("Writing output")
    evaluation.print_evaluation()
    output_path = DATA_DIR / (files[i].stem + ".csv")
    prediction.save_predictions(y_pred, output_path)
