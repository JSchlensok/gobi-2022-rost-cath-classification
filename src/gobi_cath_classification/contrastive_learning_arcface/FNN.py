from __future__ import annotations
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

from ..pipeline.model_interface import ModelInterface, Prediction
from ..pipeline.utils.torch_utils import get_device

device = get_device()


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        # TODO tune number of layers
        # TODO tune embedding size
        self.fnn = nn.Sequential(nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 128))

    @classmethod
    def from_file(cls, file: Path) -> FNN:
        model = FNN()
        model.fnn.load_state_dict(torch.load(file))
        return model

    def forward(self, x):
        return self.fnn(x)
