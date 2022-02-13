import ray
import torch
from ray import tune
import numpy as np

from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
)
from gobi_cath_classification.pipeline.data_loading import (
    load_data,
    DATA_DIR,
    scale_dataset,
)
from gobi_cath_classification.scripts_charlotte import torch_utils
from gobi_cath_classification.scripts_charlotte.models import (
    RandomForestModel,
    NeuralNetworkModel,
    GaussianNaiveBayesModel,
)


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
        eval_dict = evaluate(
            y_true=data_set.y_val,
            y_pred=y_pred_val,
            class_names_training=data_set.all_labels_train_sorted,
        )
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
                ]
            ),
        },
    )

    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
