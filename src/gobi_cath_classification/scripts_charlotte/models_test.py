import numpy as np
import torch

from gobi_cath_classification.pipeline.model_interface import ModelInterface
from gobi_cath_classification.scripts_charlotte.models import NeuralNetworkModel
from gobi_cath_classification.pipeline.torch_utils import get_device


class TestNeuralNetwork:
    def test_training(self):
        num_features = 1024
        class_names = ["1.200.45.10", "3.20.25.40", "3.200.10.75"]
        random_seed = 42

        model: ModelInterface = NeuralNetworkModel(
            lr=0.1,
            layer_sizes=[num_features],
            class_names=class_names,
            class_weights=torch.Tensor(np.array([0.5, 1, 0.3])),
            batch_size=32,
            optimizer="adam",
            rng=np.random.RandomState(random_seed),
            random_seed=random_seed,
        )

        num_data = 200
        embeddings = np.random.randn(num_data, num_features)
        labels = [
            "1.200.45.10"
            if embedding_vector[0] > 0
            else ("3.20.25.40" if embedding_vector[1] > 0 else "3.200.10.75")
            for embedding_vector in embeddings
        ]

        device = get_device()
        print(f"device = {device}")
        embeddings_tensor = torch.from_numpy(embeddings).to(device)

        for _ in range(20):
            model.train_one_epoch(
                embeddings=embeddings,
                embeddings_tensor=embeddings_tensor,
                labels=labels,
                sample_weights=None,
            )

        y_pred = model.predict_proba(np.random.randn(3, num_features)).probabilities
        assert len(y_pred) == 3
