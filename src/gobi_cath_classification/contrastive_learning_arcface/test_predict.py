import torch

from .predict import transfer_annotations
from ..pipeline.utils import CATHLabel


def test_transfer_annotations():
    lookup_embeddings = [torch.tensor([1, 1, 1, 1]), torch.tensor([5, 5, 5, 5])]
    query_embedding = torch.tensor([2, 2, 2, 2])
    labels = [CATHLabel("1.1.1.1"), CATHLabel("5.5.5.5")]
    transferred_annotation = transfer_annotations(lookup_embeddings, query_embedding, labels)
    assert transferred_annotation == "1.1.1.1"
