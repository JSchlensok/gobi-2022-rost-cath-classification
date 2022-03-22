import math
import pickle
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_inverse_class_weights,
)

from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.rnn.models import RNNModel, BRNN, BRNN_embedded
from gobi_cath_classification.rnn.pipeline import load_data
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

X_train, y_train, train_labels, X_val, y_val, X_test, y_test = load_data(
    DATA_DIR,
    np.random.RandomState(42),
    without_duplicates=True
)

sample_weights = compute_inverse_sample_weights(labels=y_train)
class_weights = torch.tensor(compute_inverse_class_weights(labels=y_train))
class_names = list(set(train_labels))

model = BRNN_embedded(
    hidden_size=512,
    num_layers=1,
    class_names=class_names,
    class_weights=class_weights,
    lr=1e-3,
    batch_size=8,
)

for e in range(100):
    metrics = model.train_one_epoch(X_train, y_train, report_progress=True)
    print(f"Epoch {e + 1}")
    print(f"Avg Loss {metrics['loss_avg']}")
    print(metrics)
    if (e % 9) == 0:
        torch.save(model, (DATA_DIR / "brnn.pth"))
    torch.cuda.empty_cache()
    with torch.no_grad():
        y_pred = model.predict(X_val)
        print(evaluate(y_val, y_pred, class_names))

torch.save(model, (DATA_DIR / "brnn.pth"))
