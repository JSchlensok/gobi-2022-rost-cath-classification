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
from gobi_cath_classification.rnn.encoder import one_hot_encode, num_categories


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

        self.lstm = nn.LSTM(
            input_size=num_categories,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=len(self.class_names)).to(
            self.device
        )

        self.relu = nn.ReLU().to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.loss_function = torch.nn.CrossEntropyLoss(weight=None)

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer is not valid: {optimizer}")

    def forward(self, x):
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
        out = self.sigmoid(out)
        return out

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
            list_indices = list_perm[i : i + self.batch_size]
            tensor_indices = tensor_perm[i : i + self.batch_size]
            batch_X = [sequences[index] for index in list_indices]
            # One Hot Encoding
            batch_X = one_hot_encode(batch_X).to(self.device)
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


class LSTMTagger(nn.Module):
    def __init__(self, hidden_dim: int, class_names: List[str]):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_categories, hidden_dim, batch_first=True, bidirectional=True)
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Linear(hidden_dim, len(class_names))

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(x.size(0), x.size(1), -1))
        print(f"lstm_out: {lstm_out.size()}")
        fc_out = self.fc(lstm_out.view(x.size(0), x.size(1), -1))
        fc_scores = self.softmax(fc_out)
        print(fc_scores.size())
        return fc_scores


class BRNN(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        class_names,
        class_weights=None,
        lr=0.01,
        batch_size=200,
    ):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch_utils.get_device()
        self.batch_size = batch_size
        self.class_names = class_names

        self.lstm = nn.LSTM(
            num_categories, hidden_size, num_layers, bidirectional=True, batch_first=True
        ).to(self.device)
        self.fc = nn.Linear(hidden_size * 2, len(class_names)).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return out

    def train_one_epoch(self, sequences: List[str], labels: List[str]) -> Dict[str, float]:
        list_perm = np.random.permutation(len(sequences))
        tensor_perm = torch.tensor(list_perm, dtype=torch.long)

        y_indices = torch.tensor([self.class_names.index(label) for label in labels]).to(
            self.device
        )

        y_one_hot = 1.0 * one_hot(y_indices, num_classes=len(self.class_names))
        loss_sum = 0

        for i in range(0, len(sequences), self.batch_size):
            list_indices = list_perm[i : i + self.batch_size]
            tensor_indices = tensor_perm[i : i + self.batch_size]
            batch_X = [sequences[index] for index in list_indices]
            # One Hot Encoding
            batch_X = one_hot_encode(batch_X).to(self.device)
            batch_y = y_one_hot[tensor_indices]
            self.optimizer.zero_grad()
            y_pred = self.forward(batch_X)

            loss = self.loss_function(y_pred, batch_y)

            loss_sum += loss
            loss.backward()
            self.optimizer.step()

        print(y_pred)
        loss_avg = float(loss_sum / (math.ceil(len(sequences) / self.batch_size)))
        model_specific_metrics = {"loss_avg": loss_avg}
        return model_specific_metrics

    def predict(self, X: List[str]) -> Prediction:
        with torch.no_grad():
            y = self.forward(one_hot_encode(X).float().to(self.device))
        df = pd.DataFrame(
            y, columns=[str(label) for label in self.class_names]
        ).astype("float")
        return Prediction(probabilities=df)