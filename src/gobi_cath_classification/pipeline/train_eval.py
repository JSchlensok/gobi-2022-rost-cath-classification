import datetime
import os
from os import listdir

from pathlib import Path

import ray
import torch
import platform
import numpy as np
from ray import tune
from ray.tune import trial

from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_class_weights,
)
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR, REPO_ROOT_DIR

from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED, set_random_seeds
from gobi_cath_classification.scripts_charlotte.models import (
    RandomForestModel,
    NeuralNetworkModel,
    GaussianNaiveBayesModel,
    DistanceModel,
)
from gobi_cath_classification.scripts_david.models import SupportVectorMachine
from gobi_cath_classification.scripts_david.save_checkpoint import (
    save_configuration,
    load_configuration,
    load_results,
    remove_files,
)


def training_function(config: dict) -> None:
    # Check if checkpoint_dir is available in config --> If yes, resume training
    if "checkpoint_dir" in config.keys():
        # Mark training function to resume training
        resume_training = True
        print(f"Attempting to resume training from checkpoint {str(config['checkpoint_dir']).split('/')[-1]} ...")
        print("Reading in configuration...")
        # Get the backup-directory of the given model
        backup_dir = Path(config['checkpoint_dir'])
        backup_dir = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, "model checkpoints" / backup_dir)
        # Read in the models saved configuration
        config = load_configuration(
            checkpoint_dir=Path(backup_dir)
        )
        # Print read in config
        for key in config.keys():
            print(f"Config: {key} = {config[key]}")
    else:
        # Do not resume training, create new model
        resume_training = False
        # Default for new config value "öast_epoch"
        config["last_epoch"] = None
        # Save the models configuration

    # Find training function file by ray tune
    with tune.checkpoint_dir(step=1) as checkpoint_dir_sub:
        # Mark as new checkpoint dir
        checkpoint_dir = Path(checkpoint_dir_sub).parent
        # os.remove(checkpoint_dir_sub)
    # Save config incl last epoch
    save_configuration(checkpoint_dir=checkpoint_dir, config=config)

    # set random seeds
    random_seed = config["random_seed"]
    set_random_seeds(seed=random_seed)
    rng = np.random.RandomState(random_seed)
    print(f"rng = {rng}")

    # load data
    data_dir = DATA_DIR
    dataset = load_data(
        data_dir=data_dir,
        rng=rng,
        without_duplicates=True,
        shuffle_data=True,
        reloading_allowed=True,
    )
    dataset.scale()

    embeddings_train, y_train_labels = dataset.get_split(split="train", zipped=False)
    embeddings_train_tensor = torch.tensor(embeddings_train)
    class_names = dataset.train_labels

    print(f"len(class_names = {len(class_names)}")

    # get hyper parameters from config dict
    print(f"config = {config}")
    model_class = config["model"]["model_class"]
    num_epochs = config["model"]["num_epochs"] if "num_epochs" in config["model"].keys() else 1

    if config["class_weights"] == "none":
        sample_weights = None
        class_weights = None
    elif config["class_weights"] == "inverse":
        sample_weights = compute_inverse_sample_weights(labels=dataset.y_train)
        class_weights = compute_class_weights(labels=dataset.y_train)
    elif config["class_weights"] == "sqrt_inverse":
        sample_weights = np.sqrt(compute_inverse_sample_weights(labels=dataset.y_train))
        class_weights = np.sqrt(compute_class_weights(labels=dataset.y_train))
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
            random_seed=RANDOM_SEED,
        )

    elif model_class == RandomForestModel.__name__:
        model = RandomForestModel(max_depth=config["model"]["max_depth"])

    elif model_class == GaussianNaiveBayesModel.__name__:
        model = GaussianNaiveBayesModel()

    elif model_class == DistanceModel.__name__:
        model = DistanceModel(
            class_names=class_names,
            embeddings=dataset.X_train,
            labels=[str(y) for y in dataset.y_train],
            distance_ord=config["model"]["distance_order"],
        )

    elif model_class == SupportVectorMachine.__name__:
        model = SupportVectorMachine(
            gamma=config["model"]["gamma"],
            c=config["model"]["regularization"],
            kernel=config["model"]["kernel_function"],
            degree=config["model"]["degree"],
        )

    else:
        raise ValueError(f"Model class {model_class} does not exist.")

    # set variables for early stopping
    highest_acc_h = 0
    n_bad = 0
    n_thresh = 20
    # If training is resumed, value will be overwritten
    epoch_start = 0

    if resume_training:
        print("Reading in model and previous results...")
        # Read in model from backup_dir
        model = model.load_model_from_checkpoint(checkpoint_dir=Path(backup_dir))
        # Read in previous results
        eval_dict = load_results(checkpoint_dir=Path(backup_dir))
        highest_acc_h = eval_dict["accuracy_h"]
        epoch_start = config["last_epoch"] + 1
        tune.report(**eval_dict)
        print(
            f"model: {model}\n epoch: {config['last_epoch']}\n h-accuracy {highest_acc_h}\n eval-dict:\n{eval_dict}"
        )
        print(f"Resuming training on model {model.__class__.__name__}...")

    print(f"Training model {model.__class__.__name__}...")
    for epoch in range(epoch_start, num_epochs):
        config["last_epoch"] = epoch
        model_metrics_dict = model.train_one_epoch(
            embeddings=embeddings_train,
            embeddings_tensor=embeddings_train_tensor,
            labels=[str(label) for label in y_train_labels],
            sample_weights=sample_weights if sample_weights is not None else None,
        )

        print(f"Predicting for X_val with model {model.__class__.__name__}...")
        y_pred_val = model.predict(embeddings=dataset.X_val)

        # evaluate and save results in ray tune
        eval_dict = evaluate(
            y_true=dataset.y_val,
            y_pred=y_pred_val,
            class_names_training=dataset.train_labels,
        )
        tune.report(**eval_dict, **{f"model_{k}": v for k, v in model_metrics_dict.items()})

        # Save the model if the average accuracy has risen during the last epoch and check for early stopping
        if eval_dict["accuracy_h"] > highest_acc_h:
            highest_acc_h = eval_dict["accuracy_h"]
            n_bad = 0
            print(f"New best performance found: accuracy_h = {highest_acc_h}")
            print(f"Attempting to save {model_class} as intermediate checkpoint...")
            # Delete old model_object and save new improved model
            remove_files(checkpoint_dir=checkpoint_dir, filetype="model_object")
            model.save_checkpoint(
                save_to_dir=checkpoint_dir,
            )
            # Remove old config file
            remove_files(checkpoint_dir=checkpoint_dir, filetype="model_configuration")
            # Save updated config file
            save_configuration(checkpoint_dir=checkpoint_dir, config=config)
        else:
            n_bad += 1
            if n_bad >= n_thresh:
                break


def trial_dirname_creator(trial: trial.Trial) -> str:
    trial_dirname = f"{trial.trainable_name}_{trial.trial_id}_{str(trial.experiment_tag)}".replace(
        ": ", ", "
    )

    # max length for path under Windows = 260 characters
    operating_system = platform.system()
    if operating_system == "Windows":
        max_len_for_trial_dirname = 260 - len(trial.local_dir)
        return trial_dirname[:max_len_for_trial_dirname]
    else:
        return trial_dirname


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
        trial_dirname_creator=trial_dirname_creator,
        local_dir="checkpoint_dir",
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "checkpoint_dir": "checkpoint_dir",
            "random_seed": RANDOM_SEED,
            "class_weights": tune.choice(["none", "inverse", "sqrt_inverse"]),
            "model": tune.grid_search(
                [
                    {
                        # unique_ID_dir = Path from within "model checkpoints" folder to model
                        "checkpoint_dir": Path(
                            "2022-03-06-23-11-26-522867/GaussianNaiveBayesModel 240b9462-e290-4f63-b077-c4ea831dd294"
                        ),
                    },
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
