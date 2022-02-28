import math
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import one_hot

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import set_random_seeds


class RNNModel(nn.Module):
    def __init__(
            self,
            lr: float,
            batch_size: int,
            optimizer: str,
            class_names: List[str],
            hidden_dim: int,
            num_layers: int,
    ):
        super(RNNModel, self).__init__()
        self.device = torch_utils.get_device()

        self.batch_size = batch_size
        self.class_names = sorted(class_names)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.categories = [
            "A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    "
            "T    W    Y    V    B    Z    X".split()
        ]

        self.lstm = nn.LSTM(
            input_size=23,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=len(self.class_names)).to(
            self.device
        )

        self.relu = nn.ReLU().to(self.device)
        self.softmax = nn.Softmax()
        self.loss_function = torch.nn.CrossEntropyLoss(weight=None)

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer is not valid: {optimizer}")

    def forward(self, x: torch.tensor):
        # print(f"Input = {x.size()}")
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(self.device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # print(f"Output = {output.size()}")
        # print(f"Hidden = {hn.size()}")
        hn = hn.view(-1, self.hidden_dim)
        # print(f"Hidden2 = {hn.size()}")
        out = self.relu(hn)
        out = self.fc(out)
        # print(f"Fully connected = {out.size()}")
        out = self.softmax(out)
        return out

    def one_hot_encode(self, X: List[str]):
        max_length = np.max([len(seq) for seq in X])
        X = [seq.ljust(max_length, "Q") for seq in X]  # Pad to the left
        X = [list(seq) for seq in X]
        X = np.array(X)
        X = X.reshape(X.shape[0], -1, 1)
        encoder = OneHotEncoder(categories=self.categories, handle_unknown="ignore")
        X = np.array([encoder.fit_transform(x).toarray() for x in X])
        return Variable(torch.tensor(X))

    def train_one_epoch(
            self,
            sequences: List[str],
            labels: List[str],
            sample_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:

        list_perm = np.random.permutation(len(sequences))
        tensor_perm = torch.tensor(list_perm, dtype=torch.long)

        y_indices = torch.tensor([self.class_names.index(label) for label in labels]).to(
            self.device
        )

        y_one_hot = 1.0 * one_hot(y_indices, num_classes=len(self.class_names))
        loss_sum = 0

        for i in range(0, len(sequences), self.batch_size):
            list_indices = list_perm[i:i + self.batch_size]
            tensor_indices = tensor_perm[i:i + self.batch_size]
            batch_X = [sequences[index] for index in list_indices]
            # One Hot Encoding
            batch_X = self.one_hot_encode(batch_X).to(self.device)
            batch_y = Variable(y_one_hot[tensor_indices])
            self.optimizer.zero_grad()
            y_pred = self.forward(batch_X.float())

            loss = self.loss_function(y_pred, batch_y)

            loss_sum += loss
            loss.backward()
            self.optimizer.step()

        loss_avg = float(loss_sum / (math.ceil(len(sequences) / self.batch_size)))
        model_specific_metrics = {"loss_avg": loss_avg}
        return model_specific_metrics

    def predict(self, sequences: List[str]):
        X = self.one_hot_encode(sequences).float()
        predicted_probabilities = self.forward(X.to(self.device))
        print(predicted_probabilities)
        df = pd.DataFrame(
            predicted_probabilities, columns=[str(label) for label in self.class_names]
        ).astype(float)
        return Prediction(probabilities=df)
