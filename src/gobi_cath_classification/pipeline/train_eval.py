import os
from pathlib import Path

import ray
import torch
import numpy as np
from ray import tune
from ray.tune import trial

from gobi_cath_classification.pipeline.Evaluation.Evaluation import Evaluation
from gobi_cath_classification.pipeline.prediction import save_predictions
from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_inverse_class_weights,
)
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR, REPO_ROOT_DIR

from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.utils.torch_utils import RANDOM_SEED, set_random_seeds
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
    # Find training function file by ray tune
    with tune.checkpoint_dir(step=1) as checkpoint_dir_sub:
        # Mark as new checkpoint dir
        checkpoint_dir = Path(checkpoint_dir_sub).parent
        # os.remove(checkpoint_dir_sub)
    # Check if checkpoint_dir is available in config --> If yes, resume training
    if "checkpoint_dir" in config["model"].keys():
        # Mark training function to resume training
        resume_training = True
        print(
            f"Attempting to resume training from checkpoint {str(config['model']['checkpoint_dir']).split('/')[-1]} ..."
        )
        print("Reading in configuration...")
        # Get the backup-directory of the given model
        backup_folder = Path(config["model"]["checkpoint_dir"])
        backup_dir = os.path.join(Path(config["model"]["local_dir"]), backup_folder)
        if not os.path.isdir(backup_dir):
            raise ValueError(f"BackUp directory: {backup_dir} - not valid!")
        # Read in the models saved configuration
        config = load_configuration(checkpoint_dir=Path(backup_dir))
        save_configuration(checkpoint_dir=checkpoint_dir, config=config)
        # Print read in config
        for key in config.keys():
            print(f"Config: {key} = {config[key]}")
    else:
        # Do not resume training, create new model
        resume_training = False
        # Default for new config value "last_epoch"
        config["last_epoch"] = None
        # Save the models configuration

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
    # scale if parameter is set in config dict, if not set: default scale = True
    if "scale" not in config["model"].keys() or config["model"]["scale"]:
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
        class_weights = compute_inverse_class_weights(labels=dataset.y_train)
    elif config["class_weights"] == "sqrt_inverse":
        sample_weights = np.sqrt(compute_inverse_sample_weights(labels=dataset.y_train))
        class_weights = np.sqrt(compute_inverse_class_weights(labels=dataset.y_train))
    else:
        raise ValueError(f'Class weights do not exist: {config["class_weights"]}')

    # set model
    if model_class == NeuralNetworkModel.__name__:
        loss_weights = None
        if "loss_weights" in config["model"].keys():
            loss_weights = torch.tensor(config["model"]["loss_weights"])
        model = NeuralNetworkModel(
            lr=config["model"]["lr"],
            class_names=[str(cn) for cn in class_names],
            layer_sizes=config["model"]["layer_sizes"],
            dropout_sizes=config["model"]["dropout_sizes"],
            batch_size=config["model"]["batch_size"],
            optimizer=config["model"]["optimizer"],
            loss_function=config["model"]["loss_function"],
            loss_weights=loss_weights,
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
        for key in eval_dict.keys():
            print(f"Result: {key} = {eval_dict[key]}")
        highest_acc_h = eval_dict["accuracy_h"]
        epoch_start = eval_dict["config"]["last_epoch"] + 1
        tune.report(**eval_dict)
        print(
            f"model: {model}\n epoch: {epoch_start-1}\n h-accuracy {highest_acc_h}\n eval-dict:\n{eval_dict}"
        )
        print(f"Resuming training on model {model.__class__.__name__} on epoch {epoch_start}...")

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
        save_predictions(pred=y_pred_val, directory=checkpoint_dir, filename="predictions.csv")

        # evaluate and save results in ray tune
        evaluation = Evaluation(
            y_true=dataset.y_val,
            predictions=y_pred_val,
            train_labels=class_names,
        )
        evaluation.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
        evaluation.compute_std_err()

        eval_dict = {}
        for k, v in evaluation.eval_dict.items():
            eval_dict = {**eval_dict, **evaluation.eval_dict[k]}

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
        else:
            n_bad += 1
            if n_bad >= n_thresh:
                break

        tune.report(
            **eval_dict,
            **{f"model_{k}": v for k, v in model_metrics_dict.items()},
            **{"highest_acc_h": highest_acc_h},
        )


def trial_dirname_creator(trial: trial.Trial) -> str:
    trial_dirname = f"{trial.trainable_name}_{trial.trial_id}_{str(trial.experiment_tag)}".replace(
        ": ", ", "
    )
    # max length for path = 260 characters
    max_len_for_trial_dirname = 260 - len(trial.local_dir)
    return trial_dirname[:max_len_for_trial_dirname]


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
    # Default Path for local_dir --> defines location of ray files
    # Can be changed to any location
    local_dir = REPO_ROOT_DIR / "model checkpoints"
    print(f"local_dir = {local_dir}")

    ray.init()
    analysis = tune.run(
        training_function,
        local_dir=local_dir,
        trial_dirname_creator=trial_dirname_creator,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "random_seed": RANDOM_SEED,
            "class_weights": tune.choice(["none", "inverse", "sqrt_inverse"]),
            "model": tune.grid_search(
                [
                    {
                        # dir to model checkpoint files from local_dir
                        "checkpoint_dir": Path(
                            "training_function_2022-03-09_01-45-07\\training_function_2ad2b_00000_0_class_weights=inverse,layer_sizes=[1024],lr=1e-05,optimizer=adam,random_seed=1_2022-03-09_01-45-07"
                        ),
                        "local_dir": local_dir,
                    },
                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 100,
                        "lr": tune.choice([1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
                        "batch_size": 32,
                        "optimizer": tune.choice(["adam", "sgd"]),
                        "loss_function": tune.choice(["CrossEntropyLoss", "HierarchicalLogLoss"]),
                        "loss_weights": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                        "layer_sizes": [1024, 2048],
                        "dropout_sizes": [0.2, None],
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
