import ray
import torch
import numpy as np
from ray import tune

from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights, compute_class_weights
)
from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.rnn.models import (
    RNNModel,
)
from gobi_cath_classification.pipeline.data_loading import DATA_DIR
from gobi_cath_classification.pipeline.data.data_loading import load_data
from gobi_cath_classification.pipeline.data.Dataset import Dataset


def training_function(config: dict) -> None:
    # set random seeds
    random_seed = config["random_seed"]
    rng = np.random.RandomState(random_seed)
    print(f"rng = {rng}")

    # load data
    data_dir = DATA_DIR
    data_set = load_data(data_dir=data_dir, without_duplicates=True, shuffle_data=True, rng=rng)
    sequences_train = data_set.train_seqs

    y_train_labels = data_set.y_train

    class_names = data_set.train_labels
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
    if model_class == RNNModel.__name__:
        model = RNNModel(
            lr=config["model"]["lr"],
            class_names=class_names,
            batch_size=config["model"]["batch_size"],
            optimizer=config["model"]["optimizer"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
        )
    else:
        raise ValueError(f"Model class {model_class} does not exist.")

    # set variables for early stopping
    highest_acc_h = 0
    n_bad = 0
    n_thresh = 20

    print(f"Training model {model.__class__.__name__}...")
    for epoch in range(num_epochs):
        model_metrics_dict = model.train_one_epoch(
            sequences=sequences_train,
            labels=y_train_labels,
            sample_weights=None,
        )

        print(f"Predicting for X_val with model {model.__class__.__name__}...")
        y_pred_val = model.predict(data_set.val_seqs)

        # evaluate and save results in ray tune
        eval_dict = evaluate(
            y_true=data_set.y_val,
            y_pred=y_pred_val,
            class_names_training=data_set.train_labels,
        )
        tune.report(**eval_dict, **{f"model_{k}": v for k, v in model_metrics_dict.items()})


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
            "random_seed": tune.grid_search([1]),
            "class_weights": tune.grid_search(["none"]),
            "model": {
                "model_class": RNNModel.__name__,
                "num_epochs": 30,
                "lr": tune.grid_search([0.1]),
                "batch_size": 100,
                "optimizer": tune.choice(["adam"]),
                "hidden_dim": tune.grid_search([64, 128, 1024]),
                "num_layers": 1,
            },
        },
        progress_reporter=reporter,
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
