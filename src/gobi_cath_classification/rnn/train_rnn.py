import ray
import torch
import numpy as np
from ray import tune

from gobi_cath_classification.pipeline.Evaluation.Evaluation import Evaluation
from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_inverse_class_weights,
)
from gobi_cath_classification.pipeline.utils.torch_utils import RANDOM_SEED
from gobi_cath_classification.rnn.models import (
    RNNModel,
)
from gobi_cath_classification.rnn.models import RNNModel, BRNN
from gobi_cath_classification.pipeline.data.data_loading import DATA_DIR, load_data
from gobi_cath_classification.pipeline.data.Dataset import Dataset


def training_function(config: dict) -> None:
    # set random seeds
    random_seed = config["random_seed"]
    rng = np.random.RandomState(random_seed)
    print(f"rng = {rng}")

    # load data
    data_dir = DATA_DIR
    dataset = load_data(
        DATA_DIR,
        np.random.RandomState(42),
        without_duplicates=True,
        load_strings=True,
        reloading_allowed=True,
    )
    X_train, y_train_labels = dataset.get_split("train", x_encoding="string", zipped=False)

    class_names = dataset.train_labels
    print(f"len(class_names) = {len(class_names)}")

    # get hyperparameters from config dict
    print(f"config = {config}")
    num_epochs = config["model"]["num_epochs"]
    model_class = config["model"]["model_class"]

    if config["class_weights"] == "none":
        sample_weights = None
        class_weights = None
    elif config["class_weights"] == "inverse":
        sample_weights = compute_inverse_sample_weights(labels=dataset.y_train)
        class_weights = compute_inverse_class_weights(labels=dataset.y_train)
    elif config["class_weights"] == "sqrt_inverse":
        sample_weights = np.sqrt(compute_inverse_sample_weights(labels=dataset.y_train))
        class_weights = np.sqrt(compute_inverse_class_weights(labels=dataset.y_train))
    else:
        raise ValueError(f'Class weights do not exist: {config["class_weights"]}')

    # set model
    if model_class == BRNN.__name__:
        model = BRNN(
            lr=config["model"]["lr"],
            class_names=class_names,
            batch_size=config["model"]["batch_size"],
            hidden_size=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            class_weights=torch.Tensor(class_weights) if class_weights is not None else None,
        )
    else:
        raise ValueError(f"Model class {model_class} does not exist.")

    X_val, y_val = dataset.get_split("val", "string", False)
    print(f"Training model {model.__class__.__name__}...")
    for epoch in range(num_epochs):
        model_metrics_dict = model.train_one_epoch(sequences=X_train, labels=y_train_labels)

        print(f"Predicting for X_val with model {model.__class__.__name__}...")
        y_pred_val = model.predict(X_val)

        # evaluate and save results in ray tune
        evaluation = Evaluation(
            y_true=y_val,
            predictions=y_pred_val,
            train_labels=class_names,
        )
        evaluation.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
        evaluation.compute_std_err()

        eval_dict = {}
        for k, v in evaluation.eval_dict.items():
            eval_dict = {**eval_dict, **evaluation.eval_dict[k]}
        print(f"eval_dict = {eval_dict}")

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

    # Where ever i save my ray results
    local_dir = "WHERE I WANT TO SAVE MY RAY RESULTS AND CHECKPOINTS"

    ray.init()
    analysis = tune.run(
        training_function,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "random_seed": tune.grid_search([1]),
            "class_weights": tune.choice(["inverse"]),
            "model": {
                "model_class": BRNN.__name__,
                "num_epochs": 60,
                "lr": tune.grid_search([0.01, 0.001, 0.0001]),
                "batch_size": tune.grid_search([32, 64]),
                "optimizer": tune.choice(["adam"]),
                "hidden_dim": tune.grid_search([1024]),
                "num_layers": 1,
            },
        },
        progress_reporter=reporter,
        local_dir=local_dir,
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
