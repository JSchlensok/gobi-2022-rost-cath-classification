import numpy as np
import pandas as pd
import torch

from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.pipeline.model_interface import ModelInterface, Prediction
from gobi_cath_classification.pipeline.utils import CATHLabel
from gobi_cath_classification.scripts_charlotte.models import (
    NeuralNetworkModel,
    DistanceModel,
    mean_squared_error,
    H_to_level_matrix,
    log_loss,
    compute_predictions_for_ensemble_model,
    HierarchicalLoss,
)
from gobi_cath_classification.pipeline.utils.torch_utils import get_device


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
        label_1, label_2, label_3 = 0, 0, 0
        for l in labels:
            if l == "1.200.45.10":
                label_1 += 1
            elif l == "3.20.25.40":
                label_2 += 1
            else:
                label_3 += 1

        class_weights = torch.tensor([label_1 / num_data, label_2 / num_data, label_3 / num_data])
        print(f"class_weights = {class_weights}")

        models = [
            NeuralNetworkModel(
                lr=0.1,
                layer_sizes=[num_features],
                dropout_sizes=[None],
                class_names=class_names,
                class_weights=class_weights,
                batch_size=32,
                optimizer="adam",
                loss_function="HierarchicalLogLoss",
                loss_weights=torch.Tensor([0.0, 0.0, 0.0, 1.0]),
                # loss_weights=torch.Tensor([1 / 10, 2 / 10, 3 / 10, 4 / 10]),
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


class TestHierarchicalLoss:
    def test_hierarchical_log_loss(self):
        class_names = ["1.1.1.1", "1.1.1.2", "1.1.2.3"]
        class_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])

        loss_function = HierarchicalLoss(
            class_names=class_names,
            class_weights=class_weights,
            hierarchical_weights=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            device="cpu",
            loss_function=log_loss,
        )
        y_true = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        y_pred = torch.tensor(
            [
                [0.0, 0.1, 0.9],
                [1.0, 0.0, 0.0],
                [0.2, 0.8, 0.0],
            ]
        )
        loss = loss_function(y_pred=y_pred, y_true=y_true)
        print(f"loss = {loss}")
        assert torch.allclose(loss, torch.tensor([0.328504066972036 / 9]))


def test_H_to_level_matrix():
    class_names = ["1.1.1.1", "2.2.3.3", "2.3.3.3", "2.3.4.4", "2.3.4.5"]

    matrix_C = torch.transpose(
        torch.Tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1],
            ]
        ),
        0,
        1,
    )
    matrix_A = torch.transpose(
        torch.Tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1],
            ]
        ),
        0,
        1,
    )
    matrix_T = torch.transpose(
        torch.Tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ]
        ),
        0,
        1,
    )
    matrix_H = torch.eye(n=len(class_names))

    assert torch.equal(matrix_C, H_to_level_matrix(class_names=class_names, level="C"))
    assert torch.equal(matrix_A, H_to_level_matrix(class_names=class_names, level="A"))
    assert torch.equal(matrix_T, H_to_level_matrix(class_names=class_names, level="T"))
    assert torch.equal(matrix_H, H_to_level_matrix(class_names=class_names, level="H"))


def test_mean_squared_error():
    y_true = torch.Tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    y_pred = torch.Tensor([[0.0, 0.1, 0.9], [0.0, 0.1, 0.9]])
    mse = mean_squared_error(y_pred=y_pred, y_true=y_true)

    assert torch.allclose(mse, torch.tensor([0.04]))


def test_log_loss():
    y_true = torch.Tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    y_pred = torch.Tensor([[0.0, 0.1, 0.9], [0.0, 0.1, 0.9]])
    loss = log_loss(y_pred=y_pred, y_true=y_true)
    print(f"loss = {loss}")

    assert torch.allclose(loss, torch.tensor([0.21072109043598175 / 2]))


def test_compute_predictions_for_ensemble_model():
    col_names = ["1.200.45.10", "3.20.25.40", "3.200.10.75"]
    pred1 = Prediction(pd.DataFrame(data=np.array([[1, 2, 3], [2, 3, 4]]), columns=col_names))
    pred2 = Prediction(pd.DataFrame(data=np.array([[4, 5, 6], [5, 6, 7]]), columns=col_names))

    ensemble_pred = compute_predictions_for_ensemble_model(
        predictions_from_models=[pred1, pred2], weights=np.array([0.8, 0.2])
    )
    np.testing.assert_allclose(
        actual=ensemble_pred.probabilities, desired=np.array([[1.6, 2.6, 3.6], [2.6, 3.6, 4.6]])
    )
