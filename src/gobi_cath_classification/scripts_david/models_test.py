# Import basic, important functionalities
import numpy as np
import torch

# Import functionalities from different packages in the directory
from gobi_cath_classification.pipeline.model_interface import ModelInterface
from gobi_cath_classification.pipeline.utils.torch_utils import get_device
from .models import SupportVectorMachine


class TestSupportVectorMachine:
    ########################################################################################
    # CLASS NAME        : TestSupportVectorMachine
    # IMPLEMENTS        : nothing
    # DESCRIPTION       : Class to implement a testinstance for Support Vector Machines
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 18.02.2022
    # UPDATE            : ---
    ########################################################################################
    def test_training(self):
        num_features = 1024
        class_names = ["1.200.45.10", "3.20.25.40", "3.200.10.75"]
        random_seed = 42

        model: ModelInterface = SupportVectorMachine()

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

        for _ in range(1):
            model.train_one_epoch(
                embeddings=embeddings,
                embeddings_tensor=embeddings_tensor,
                labels=labels,
                sample_weights=None,
            )

        y_pred = model.predict(np.random.randn(3, num_features)).probabilities
        assert len(y_pred) == 3
