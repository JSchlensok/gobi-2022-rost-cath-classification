import math
import pickle
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

from gobi_cath_classification.pipeline.Evaluation import Evaluation
from gobi_cath_classification.rnn.pipeline import load_data
from gobi_cath_classification.pipeline.data.data_loading import DATA_DIR
from gobi_cath_classification.pipeline import prediction

_, _, train_labels, _, _, X_test, y_test, X_tmp, y_tmp = load_data(
    data_dir=DATA_DIR, load_tmp_holdout_set=True
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


def print_evaluation(y_true, predictions, name, save_file):
    evaluation = Evaluation(
        y_true=y_true, predictions=predictions, train_labels=class_names, model_name=name
    )
    print("Computing scores...")
    evaluation.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
    # print("Computing error...")
    # evaluation.compute_std_err()

    print("Writing output")
    evaluation.print_evaluation()
    prediction.save_predictions(y_pred, save_file)


for i in range(len(models)):
    model = models[i]
    model_name = files[i].stem
    print(f"Predicting for model{model_name} on the test set")
    y_pred = model.predict(X_test)

    output_path = DATA_DIR / (model_name + "_test.csv")
    print_evaluation(y_test, y_pred, model_name, output_path)

    print(f"Predicting for model {model_name} on the temporal holdout set")
    print(f"Temporal Holdout Set :{len(X_tmp)} \n {X_tmp}")
    print(f"Test Set :{len(X_test)} \n {X_test}")
    y_pred = model.predict(X_tmp)

    output_path = DATA_DIR / (model_name + "_tmph.csv")
    print_evaluation(y_tmp, y_pred, model_name, output_path)
