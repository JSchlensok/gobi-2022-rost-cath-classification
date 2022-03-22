import math
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.nn.functional import one_hot

from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds
from gobi_cath_classification.rnn.encoder import one_hot_encode, num_categories, pad_embeddings
from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.utils.torch_utils import set_random_seeds

categories = [
    "A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    "
    "T    W    Y    V    B    Z    X".split()
]


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
            batch_first=True,
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
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
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
            batch_y = y_one_hot[tensor_indices]
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
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out_forward = out[:, -1, : self.hidden_size]
        out_backward = out[:, 0, self.hidden_size :]
        out = torch.cat((out_forward, out_backward), dim=1)
        out = self.fc(out)
        out = self.softmax(out)

        return out

    def train_one_epoch(
        self, sequences: List[str], labels: List[str], report_progress=False
    ) -> Dict[str, float]:
        list_perm = np.random.permutation(len(sequences))
        tensor_perm = torch.tensor(list_perm, dtype=torch.long)

        y_indices = torch.tensor([self.class_names.index(label) for label in labels]).to(
            self.device
        )

        y_one_hot = 1.0 * one_hot(y_indices, num_classes=len(self.class_names))
        counter = 0
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
            if report_progress & (counter % 18 == 0):
                print(f"\t\t{i + self.batch_size}/{len(sequences)} done")
            counter += 1

        loss_avg = float(loss_sum / (math.ceil(len(sequences) / self.batch_size)))
        model_specific_metrics = {"loss_avg": loss_avg}
        return model_specific_metrics

    def predict(self, X: List[str]) -> Prediction:
        with torch.no_grad():
            y = self.forward(one_hot_encode(X).float().to(self.device))
        df = pd.DataFrame(y, columns=[str(label) for label in self.class_names]).astype("float")
        return Prediction(probabilities=df)


class BRNN_embedded(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_layers,
            class_names,
            class_weights=None,
            lr=0.01,
            batch_size=200,
    ):
        super(BRNN_embedded, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch_utils.get_device()
        self.batch_size = batch_size
        self.class_names = class_names

        self.lstm = nn.LSTM(
            1024, hidden_size, num_layers, bidirectional=False, batch_first=True
        ).to(self.device)
        self.fc = nn.Linear(hidden_size, len(class_names)).to(self.device)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.ReLU(x[:, -1, :])
        # x = torch.cat((x[:, -1, :self.hidden_size], x[:, 0, self.hidden_size:]), dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def train_one_epoch(
            self, embeddings: List, labels: List[str], report_progress=False
    ) -> Dict[str, float]:
        list_perm = np.random.permutation(len(embeddings))
        tensor_perm = torch.tensor(list_perm, dtype=torch.long)

        y_indices = torch.tensor([self.class_names.index(label) for label in labels])

        y_one_hot = 1.0 * one_hot(y_indices, num_classes=len(self.class_names)).to(self.device)
        counter = 0
        loss_sum = 0

        for i in range(0, len(embeddings), self.batch_size):
            list_indices = list_perm[i: i + self.batch_size]
            tensor_indices = tensor_perm[i: i + self.batch_size]
            batch_X = [embeddings[index] for index in list_indices]
            # Pad the embeddings
            batch_X = pad_embeddings(batch_X).to(self.device)
            batch_y = y_one_hot[tensor_indices]
            self.optimizer.zero_grad()
            y_pred = self.forward(batch_X)

            loss = self.loss_function(y_pred, batch_y)

            loss_sum += loss
            loss.backward()
            self.optimizer.step()

            if report_progress & (counter % 300 == 0):
                print(f"\t\t{i + self.batch_size}/{len(embeddings)} done")
            counter += 1

            del loss, y_pred

        loss_avg = float(loss_sum / (math.ceil(len(embeddings) / self.batch_size)))
        model_specific_metrics = {"loss_avg": loss_avg}
        return model_specific_metrics

    def predict(self, X: List) -> Prediction:
        with torch.no_grad():
            y = self.forward(pad_embeddings(X).to(self.device))
        df = pd.DataFrame(y, columns=[str(label) for label in self.class_names]).astype("float")
        return Prediction(probabilities=df)

