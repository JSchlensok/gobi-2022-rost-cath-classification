import math
import pickle
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_inverse_class_weights,
)
from torch import nn, optim
from torch.nn.functional import one_hot
from sklearn.preprocessing import OneHotEncoder

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.Evaluation import Evaluation
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds
from gobi_cath_classification.rnn.models import RNNModel, BRNN, one_hot_encode
from gobi_cath_classification.pipeline.data.data_loading import DATA_DIR, load_data
from gobi_cath_classification.pipeline.data.Dataset import Dataset

# dataset = pickle.load(
#     open(DATA_DIR / "serialized_dataset_no-duplicates_full_with_strings.pickle", "rb")
# )
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
device = torch_utils.get_device()
print(f"device = {device}")

if torch.cuda.is_available():
    resources_per_trial = {"gpu": 1}
else:
    resources_per_trial = {"cpu": 1}

dataset = load_data(
    DATA_DIR,
    np.random.RandomState(42),
    without_duplicates=True,
    load_strings=True,
    reloading_allowed=True,
    load_tmp_holdout_set=False,
)

sample_weights = compute_inverse_sample_weights(labels=dataset.y_train)
class_weights = torch.tensor(compute_inverse_class_weights(labels=dataset.y_train))
class_names = dataset.train_labels

model = BRNN(
    hidden_size=1600,
    num_layers=1,
    class_names=class_names,
    class_weights=class_weights,
    lr=1e-4,
    batch_size=32,
)
X_train, y_train_labels = dataset.get_split("train", x_encoding="string", zipped=False)
X_val, y_val = dataset.get_split("val", x_encoding="string", zipped=False)

for e in range(100):
    metrics = model.train_one_epoch(X_train, y_train_labels, report_progress=True)
    print(f"Epoch {e + 1}")
    print(f"Avg Loss {metrics['loss_avg']}")
    print(metrics)
    if (e % 9) == 0:
        torch.save(model, (DATA_DIR / "brnn.pth"))
    torch.cuda.empty_cache()
    with torch.no_grad():
        y_pred = model.predict(X_val)

        evaluation = Evaluation(
            y_true=y_val, predictions=y_pred, train_labels=class_names, model_name="BRNN"
        )  # can be changed
        evaluation.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
        evaluation.compute_std_err()

        eval_dict = {}
        for k, v in evaluation.eval_dict.items():
            eval_dict = {**eval_dict, **evaluation.eval_dict[k]}
        print(f"eval_dict = {eval_dict}")

torch.save(model, (DATA_DIR / "brnn.pth"))
