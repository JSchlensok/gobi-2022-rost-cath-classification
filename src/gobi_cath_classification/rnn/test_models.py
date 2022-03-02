import math
import pickle
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_class_weights,
)
from torch import nn, optim
from torch.nn.functional import one_hot
from sklearn.preprocessing import OneHotEncoder

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import set_random_seeds
from gobi_cath_classification.rnn.models import RNNModel, BRNN, one_hot_encode
from gobi_cath_classification.pipeline.data.data_loading import load_data
from gobi_cath_classification.pipeline.data_loading import DATA_DIR
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
)

sample_weights = compute_inverse_sample_weights(labels=dataset.y_train)
class_weights = torch.tensor(compute_class_weights(labels=dataset.y_train))
class_names = dataset.train_labels

model = BRNN(
    hidden_size=256,
    num_layers=1,
    class_names=class_names,
    class_weights=class_weights,
    lr=1e-5,
    batch_size=32,
)
X_train, y_train_labels = dataset.get_split("train", x_encoding="string", zipped=False)


for e in range(100):
    metrics = model.train_one_epoch(X_train, y_train_labels, report_progress=True)
    print(f"Epoch {e + 1}")
    print(f"Avg Loss {metrics['loss_avg']}")
    print(metrics)
    if (e % 9) == 0:
        torch.save(model, (DATA_DIR / "brnn.pth"))

torch.save(model, (DATA_DIR / "brnn.pth"))
