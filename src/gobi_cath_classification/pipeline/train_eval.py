import ray
import uuid
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
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR

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
    save_model_configuration,
    save_model_results,
    save_model,
    load_configuration,
    load_results,
    load_model,
)


def training_function(config: dict) -> None:
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

    # get hyperparameters from config dict
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

    # Create a unique ID to later identify the model later
    index_uniqueID = uuid.uuid4()
    print(f"CURRENT MODEL'S ASSIGNED UNIQUE ID - {index_uniqueID}")

    print(f"Training model {model.__class__.__name__}...")

    # Make eval_dict available outside the for loop
    eval_dict = None
    highest_avg_accuracy = 0

    # Save the model configuration before running training
    save_model_configuration(model_class=model_class, unique_ID=index_uniqueID, dict_config=config)

    for epoch in range(num_epochs):
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

        # Save the model if the average accuracy has risen during the last epoch
        if eval_dict["accuracy_avg"] > highest_avg_accuracy:
            highest_avg_accuracy = eval_dict["accuracy_avg"]
            # UPDATE - David Mauder 01.03.2022
            # Attempting to save the current model state after one training epoch
            save_model(model=model,
                       model_class=model_class,
                       unique_ID=index_uniqueID,
                       epoch=epoch)
            # Attempting to save the current model results after evaluation
            save_model_results(model_class=model_class, unique_ID=index_uniqueID, eval_dict=eval_dict, epoch=epoch)


def resume_training(config: dict) -> None:
    ########################################################################################
    # FUNCTION NAME     : resume_training()
    # INPUT PARAMETERS  : config: dict
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : resumes training of a previously saved model state
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 03.03.2022
    # UPDATE            : ---
    ########################################################################################
    unique_ID = config["unique_ID"]
    print(f"Attempting to resume training for ID {unique_ID}...")
    print("Reading in configuration...")
    config = load_configuration(unique_ID=unique_ID)
    random_seed = int(config["random_seed"])
    set_random_seeds(seed=random_seed)
    rng = np.random.RandomState(random_seed)
    print(f"rng = {rng}")

    data_dir = DATA_DIR
    data_set = scale_dataset(
        load_data(
            data_dir=data_dir,
            without_duplicates=True,
            shuffle_data=True,
            rng=rng,
        )
    )

    class_names = data_set.all_labels_train_sorted
    print(f"len(class_names = {len(class_names)}")

    # Hyperparameters
    print(f"config = {config}")
    num_epochs = config["num_epochs"]
    model_class = config["model_class"]

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

    print("Reading in model and previous results...")
    model, epoch, uniqueID = load_model(unique_ID=unique_ID)
    eval_dict, avg_accuracy = load_results(unique_ID=unique_ID)
    tune.report(**eval_dict)
    print(f"unique ID: {uniqueID}, model: {model}, epoch: {epoch}, avg-accuracy {avg_accuracy}, eval-dict:\n{eval_dict}")

    embeddings_train = data_set.X_train
    embeddings_train_tensor = torch.tensor(embeddings_train)

    y_train_labels = data_set.y_train

    print(f"Resuming training on model {model.__class__.__name__}...")

    # Make eval_dict available outside the for loop
    for epoch in range(epoch + 1, int(num_epochs)):
        model.train_one_epoch(
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
        tune.report(**eval_dict)

        # Save the model if the average accuracy has risen during the last epoch
        if eval_dict["accuracy_avg"] > avg_accuracy:
            avg_accuracy = eval_dict["accuracy_avg"]
            # UPDATE - David Mauder 01.03.2022
            # Attempting to save the current model state after one training epoch
            save_model(model=model,
                       model_class=model_class,
                       unique_ID=uniqueID,
                       epoch=epoch)
            # Attempting to save the current model results after evaluation
            save_model_results(model_class=model_class, unique_ID=uniqueID, eval_dict=eval_dict, epoch=epoch)


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
