from typing import List, Dict

import ray
import torch
from ray import tune
import numpy as np
from sklearn.metrics import accuracy_score

from gobi_cath_classification.pipeline.model_interface import Prediction
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
)
from gobi_cath_classification.pipeline.data_loading import (
    load_data,
    check_if_cath_level_is_valid,
    label_for_level,
    DATA_DIR,
    scale_dataset,
)
from gobi_cath_classification.scripts_charlotte import torch_utils
from gobi_cath_classification.scripts_charlotte.models import (
    RandomForestModel,
    NeuralNetworkModel,
    GaussianNaiveBayesModel,
)


def accuracy_for_level(y_true: List[str], y_pred: List[str], cath_level: str) -> float:
    # TODO add parameter: keep_testset_entries_if_label_not_in_training
    check_if_cath_level_is_valid(cath_level=cath_level)

    y_true_for_level = [label_for_level(label=label, cath_level=cath_level) for label in y_true]
    y_pred_for_level = [label_for_level(label=label, cath_level=cath_level) for label in y_pred]

    return accuracy_score(y_true=y_true_for_level, y_pred=y_pred_for_level)


def evaluate(y_true: List[str], y_pred: Prediction) -> Dict[str, float]:
    y_proba = y_pred.probabilities
    y_labels = y_pred.argmax_labels()
    eval_dict = {
        "accuracy_c": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="C"),
        "accuracy_a": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="A"),
        "accuracy_t": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="T"),
        "accuracy_h": accuracy_for_level(y_true=y_true, y_pred=y_labels, cath_level="H"),
        # "roc_ovr_micro_avg": roc_auc_score(y_true=y_true, y_score=y_proba, average="micro",
        #                                    multi_class="ovr"),
        # "roc_ovr_macro_avg": roc_auc_score(y_true=y_true, y_score=y_proba, average="macro",
        #                                    multi_class="ovr"),
    }
    eval_dict["accuracy_avg"] = (
        eval_dict["accuracy_c"]
        + eval_dict["accuracy_a"]
        + eval_dict["accuracy_t"]
        + eval_dict["accuracy_h"]
    ) / 4
    return eval_dict


def training_function(config: dict) -> None:
    data_dir = DATA_DIR
    data_set = scale_dataset(
        load_data(
            data_dir=data_dir,
            without_duplicates=True,
            shuffle_data=True,
        )
    )

    class_names = data_set.all_labels_train_sorted
    print(f"len(class_names = {len(class_names)}")

    # Hyperparameters
    print(f"config = {config}")

    num_epochs = config["model"]["num_epochs"]
    model_class = config["model"]["model_class"]

    if config["class_weights"] == "none":
        sample_weights = None
    elif config["class_weights"] == "inverse":
        sample_weights = compute_inverse_sample_weights(labels=data_set.y_train)
    elif config["class_weights"] == "sqrt_inverse":
        sample_weights = np.sqrt(compute_inverse_sample_weights(labels=data_set.y_train))
    else:
        raise ValueError(f'Class weights do not exist: {config["class_weights"]}')

    if model_class == NeuralNetworkModel.__name__:
        model = NeuralNetworkModel(
            class_names=class_names,
            layer_sizes=config["model"]["layer_sizes"],
            batch_size=config["model"]["batch_size"],
            optimizer=config["model"]["optimizer"],
            lr=config["model"]["lr"],
        )

    elif model_class == RandomForestModel.__name__:
        model = RandomForestModel(max_depth=config["model"]["max_depth"])

    elif model_class == GaussianNaiveBayesModel.__name__:
        model = GaussianNaiveBayesModel()

    else:
        raise ValueError(f"Model class {model_class} does not exist.")

    embeddings_train = data_set.X_train
    embeddings_train_tensor = torch.tensor(embeddings_train)

    y_train_labels = data_set.y_train

    print(f"Training model {model.__class__.__name__}...")
    for epoch in range(num_epochs):
        model.train_one_epoch(
            embeddings=embeddings_train,
            embeddings_tensor=embeddings_train_tensor,
            labels=y_train_labels,
            sample_weights=sample_weights if sample_weights is not None else None,
        )

        print(f"Predicting for X_val with model {model.__class__.__name__}...")
        y_pred_val = model.predict_proba(embeddings=data_set.X_val)

        # evaluate and save results in ray tune
        eval_dict = evaluate(y_true=data_set.y_val, y_pred=y_pred_val)
        tune.report(**eval_dict)


def main():
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = torch_utils.get_device()
    print(f"device = {device}")

    ray.init()
    analysis = tune.run(
        training_function,
        num_samples=1,
        config={
            "class_weights": tune.choice(["none", "inverse", "sqrt_inverse"]),
            "model": tune.grid_search(
                [
                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 100,
                        "lr": tune.choice([1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
                        "batch_size": 32,
                        "optimizer": tune.choice(["adam", "sgd"]),
                        "layer_sizes": tune.choice(
                            [
                                [1024],
                                [1024, 128],
                                [1024, 256],
                            ]
                        ),
                    },
                    {
                        "model_class": GaussianNaiveBayesModel.__name__,
                        "num_epochs": 1,
                    },
                    #     {
                    #         "model_class": RandomForestModel.__name__,
                    #         "num_epochs": 1,
                    #         "max_depth": tune.choice([1, 2, 3]),
                    #     },
                ]
            ),
        },
    )

    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
