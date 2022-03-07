import numpy as np
import torch

from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.pipeline.model_interface import ModelInterface
from gobi_cath_classification.pipeline.utils import CATHLabel
from gobi_cath_classification.scripts_charlotte.models import (
    NeuralNetworkModel,
    DistanceModel,
    _get_predictions_for_level,
    hierarchical_loss,
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
                dropout_sizes=[None],
                class_names=class_names,
                class_weights=torch.Tensor(np.array([0.5, 1, 0.3])),
                batch_size=32,
                optimizer="adam",
                rng=np.random.RandomState(random_seed),
                random_seed=random_seed,
            ),
            DistanceModel(
                class_names=class_names, embeddings=embeddings, labels=labels, distance_ord=1
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

            y_pred = model.predict(np.random.randn(5, num_features))
            assert len(y_pred.probabilities) == 5

            y_true = [
                CATHLabel("1.200.45.10"),
                CATHLabel("3.20.25.40"),
                CATHLabel("3.200.10.75"),
                CATHLabel("3.20.25.40"),
                CATHLabel("3.200.10.75"),
            ]

            eval_dict = evaluate(
                y_true=y_true,
                y_pred=y_pred,
                class_names_training=[CATHLabel(cn) for cn in class_names],
            )
            print(f"eval_dict = {eval_dict}")


def test_get_predictions_for_level_C():
    labels = ["1.10.20.20", "1.35.20.20", "2.400.20.20"]
    y_pred = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.6, 0.1],
        ]
    )
    y_pred_level_C = _get_predictions_for_level(cath_level="C", y_pred=y_pred, labels=labels)
    assert np.allclose(y_pred_level_C, np.array([[0.3, 0.7], [0.9, 0.1]]))
    assert np.allclose(y_pred_level_C.shape, (2, 2))


def test_hierarchical_loss():
    labels_in_train = ["1.10.20.20", "2.35.20.20", "2.400.30.10"]
    y_true = torch.from_numpy(
        np.array(
            [
                [0.0, 0.0, 1.0],
            ]
        )
    )

    y_pred = torch.from_numpy(
        np.array(
            [
                [0.0, 0.0, 3.0],
            ]
        )
    )

    loss_function = torch.nn.CrossEntropyLoss()

    loss = hierarchical_loss(
        y_pred=y_pred,
        y_true=y_true,
        labels=labels_in_train,
        weights=[0.25, 0.25, 0.25, 0.25],
        loss_function=loss_function,
    )
    print(f"\nloss = {loss}")
