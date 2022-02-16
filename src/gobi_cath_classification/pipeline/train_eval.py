import os
import random

import numpy as np
import ray
import torch
from ray import tune

from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_class_weights,
)
from gobi_cath_classification.pipeline.data_loading import (
    load_data,
    DATA_DIR,
    scale_dataset,
)
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED, set_random_seeds
from gobi_cath_classification.scripts_charlotte.models import (
    RandomForestModel,
    NeuralNetworkModel,
    GaussianNaiveBayesModel,
)


def training_function(config: dict) -> None:
    # set random seeds
    random_seed = config["random_seed"]
    set_random_seeds(seed=random_seed)
    rng = np.random.RandomState(random_seed)
    print(f"rng = {rng}")

    # load data
    data_dir = DATA_DIR
    data_set = scale_dataset(
        load_data(
            data_dir=data_dir,
            without_duplicates=True,
            shuffle_data=True,
            rng=rng,
        )
    )
    embeddings_train = data_set.X_train
    embeddings_train_tensor = torch.tensor(embeddings_train)

    y_train_labels = data_set.y_train

    class_names = data_set.all_labels_train_sorted
    print(f"len(class_names) = {len(class_names)}")

    # get hyperparameters from config dict
    print(f"config = {config}")
    num_epochs = config["model"]["num_epochs"]
    model_class = config["model"]["model_class"]

    if config["class_weights"] == "none":
        sample_weights = None
        class_weights = None
    elif config["class_weights"] == "inverse":
        sample_weights = compute_inverse_sample_weights(labels=data_set.y_train)
        class_weights = compute_class_weights(labels=data_set.y_train)
    elif config["class_weights"] == "sqrt_inverse":
        sample_weights = np.sqrt(compute_inverse_sample_weights(labels=data_set.y_train))
        class_weights = np.sqrt(compute_class_weights(labels=data_set.y_train))
    else:
        raise ValueError(f'Class weights do not exist: {config["class_weights"]}')

    # set model
    if model_class == NeuralNetworkModel.__name__:
        model = NeuralNetworkModel(
            lr=config["model"]["lr"],
            class_names=class_names,
            layer_sizes=config["model"]["layer_sizes"],
            batch_size=config["model"]["batch_size"],
            optimizer=config["model"]["optimizer"],
            class_weights=torch.Tensor(class_weights) if class_weights is not None else None,
            rng=rng,
            random_seed=random_seed,
        )

    elif model_class == RandomForestModel.__name__:
        model = RandomForestModel(max_depth=config["model"]["max_depth"])

    elif model_class == GaussianNaiveBayesModel.__name__:
        model = GaussianNaiveBayesModel()

    else:
        raise ValueError(f"Model class {model_class} does not exist.")

    # set variables for early stopping
    highest_acc_h = 0
    n_bad = 0
    n_thresh = 20

    print(f"Training model {model.__class__.__name__}...")
    for epoch in range(num_epochs):
        model_metrics_dict = model.train_one_epoch(
            embeddings=embeddings_train,
            embeddings_tensor=embeddings_train_tensor,
            labels=y_train_labels,
            sample_weights=sample_weights if sample_weights is not None else None,
        )

        print(f"Predicting for X_val with model {model.__class__.__name__}...")
        y_pred_val = model.predict(embeddings=data_set.X_val)

        # evaluate and save results in ray tune
        eval_dict = evaluate(
            y_true=data_set.y_val,
            y_pred=y_pred_val,
            class_names_training=data_set.all_labels_train_sorted,
        )
        tune.report(**eval_dict, **{f"model_{k}": v for k, v in model_metrics_dict.items()})

        # check for early stopping
        acc_h = eval_dict["accuracy_h"]
        if acc_h > highest_acc_h:
            highest_acc_h = acc_h
            n_bad = 0
            print(f"New best performance found: accuracy_h = {highest_acc_h}")
        else:
            n_bad += 1
            if n_bad >= n_thresh:
                break


def main():
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = torch_utils.get_device()
    print(f"device = {device}")

    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 1}
    else:
        resources_per_trial = {"cpu": 1}

    reporter = tune.CLIReporter(
        max_report_frequency=10,
        infer_limit=10,
    )

    ray.init()
    analysis = tune.run(
        training_function,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "random_seed": RANDOM_SEED,
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
                ]
            ),
        },
        progress_reporter=reporter,
    )

    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
