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
from gobi_cath_classification.rnn.models import RNNModel, LSTMTagger, BRNN, one_hot_encode
from gobi_cath_classification.pipeline.data.data_loading import load_data
from gobi_cath_classification.pipeline.data_loading import DATA_DIR
from gobi_cath_classification.pipeline.data.Dataset import Dataset

# dataset = pickle.load(
#     open(DATA_DIR / "serialized_dataset_no-duplicates_full_with_strings.pickle", "rb")
# )
dataset = load_data(
    DATA_DIR, np.random.RandomState(42), without_duplicates=True, load_strings=True, reloading_allowed=True
)
sample_weights = compute_inverse_sample_weights(labels=dataset.y_train)
class_weights = compute_class_weights(labels=dataset.y_train)
class_names = dataset.train_labels

model = BRNN(hidden_size=128, num_layers=1, class_names=class_names)
X_train, y_train_labels = dataset.get_split("train", x_encoding="string", zipped=False)

for e in range(50):
    metrics = model.train_one_epoch(X_train, y_train_labels)
    print(f"Epoch {e}")
    print(f"Avg Loss{metrics['loss_avg']}")
