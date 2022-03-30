from typing import Callable

import sklearn
import torch
from torchtyping import TensorType


class EAT:
    def __init__(self, distance: Callable, lookup: TensorType[-1, 128], query: TensorType[-1, 128]):
        self.distance = distance
        self.lookup = lookup
        self.query = query

        self.neighbor_indices = None
        self.encoded_labels = None
        self.decoded_labels = None

    def get_neighbors(self, k: int) -> None:
        distances = self.distance(self.query.float(), self.lookup.float())
        _, self.neighbor_indices = torch.topk(distances, k, largest=False)

    def transfer_labels(self, lookup_labels: TensorType[-1, torch.int64]) -> None:
        # TODO handle k>1
        self.encoded_labels = torch.take(lookup_labels, self.neighbor_indices)

    def decode_labels(self, encoder: sklearn.preprocessing.LabelEncoder) -> None:
        self.decoded_labels = encoder.inverse_transform(self.encoded_labels.cpu().flatten())
