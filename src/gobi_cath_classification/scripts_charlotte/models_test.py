import numpy as np
import torch

from gobi_cath_classification.pipeline.model_interface import ModelInterface
from gobi_cath_classification.scripts_charlotte.models import (
    NeuralNetworkModel,
    DistanceModel, distance,
)
from gobi_cath_classification.pipeline.torch_utils import get_device


class TestNeuralNetwork:
    def test_training(self):
        num_features = 1024
        class_names = ["1.200.45.10", "3.20.25.40", "3.200.10.75"]
        random_seed = 42

        num_data = 200
        embeddings = np.random.randn(num_data, num_features)
        labels = [
            "1.200.45.10"
            if embedding_vector[0] > 0
            else ("3.20.25.40" if embedding_vector[1] > 0 else "3.200.10.75")
            for embedding_vector in embeddings
        ]

        models = [
            NeuralNetworkModel(
                lr=0.1,
                layer_sizes=[num_features],
                class_names=class_names,
                class_weights=torch.Tensor(np.array([0.5, 1, 0.3])),
                batch_size=32,
                optimizer="adam",
                rng=np.random.RandomState(random_seed),
                random_seed=random_seed,
            ),
            DistanceModel(
                class_names=class_names,
                embeddings=embeddings,
                labels=labels,
                distance_ord=1
            ),
        ]
        for model in models:
            model: ModelInterface = model

            device = get_device()
            print(f"device = {device}")
            embeddings_tensor = torch.from_numpy(embeddings).to(device)

            for _ in range(1):
                model.train_one_epoch(
                    embeddings=embeddings,
                    embeddings_tensor=embeddings_tensor,
                    labels=labels,
                    sample_weights=None,
                )

            y_pred = model.predict(np.random.randn(5, num_features)).probabilities
            assert len(y_pred) == 5


def test_scratch():
    print()
    embeddings = np.array(
        [
            [1, 1, 1, 1, 1],
            [5, 5, 5, 5, 5]
        ]
    )
    X_train = np.array(
        [
            [1, 2, 3, 2, 1],
            [5, 6, 6, 5, 5],
            [5, 5, 5, 5, 5],
            [1, 1, 1, 1, 1],
            [1, 6, 9, 9, 9]
        ]
    )

    emb_tensor = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [5, 5, 5, 5, 5]
        ]
    )
    X_train_tensor = torch.tensor(
        [
            [1, 2, 3, 2, 1],
            [5, 6, 6, 5, 5],
            [5, 5, 5, 5, 5],
            [1, 1, 1, 1, 1],
            [1, 6, 9, 9, 9]
        ]
    )
    dist_ord = 1
    np_X_train = X_train_tensor.numpy()
    print(f"type(np_X_train) = {type(np_X_train)}")
    # dist = torch.cdist(emb_tensor[0], emb_tensor[0], p=2)
    pdist = torch.nn.PairwiseDistance(p=dist_ord, eps=1e-08)
    distances_t = np.array([[pdist(emb, emb_lookup) for emb_lookup in X_train_tensor] for emb in emb_tensor])

    print(f"emb_tensor[0] = {emb_tensor[0]}")
    print(f"dist = {distances_t}")
    print(f"distances_t.shape = {distances_t.shape}")

    distances = []
    for i, emb in enumerate(embeddings):
        dists = []
        for j, emb_lookup in enumerate(X_train):
            dist = distance(emb, emb_lookup, dist_ord=dist_ord)
            dists.append(dist)
        distances.append(dists)
    distances = np.array(distances)
    print(f"\ndistances = {distances}")
    print(f"distances.shape = {distances.shape}")
    assert np.allclose(distances.shape, (2, 5))

    distances_2 = []
    distances_2 = [[distance(emb, emb_lookup, dist_ord=dist_ord) for emb_lookup in X_train] for emb in embeddings]
    assert np.allclose(distances, distances_2)

