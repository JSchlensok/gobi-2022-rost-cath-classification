import numpy as np
import ray
import torch
from ray import tune
from ray.tune import trial

import platform
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.data_loading import (
    load_data,
    DATA_DIR,
    scale_dataset,
)
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED, set_random_seeds
from gobi_cath_classification.pipeline.evaluation import evaluate
from gobi_cath_classification.scripts_finn.baseline_models import RandomBaseline, ZeroRate


def training_function(config: dict) -> None:

    # set random seeds
    random_seed = config["random_seed"]
    set_random_seeds(seed=random_seed)
    rng = np.random.RandomState(random_seed)
    print(f"rng = {rng}")

    # load in the data
    data_dir = DATA_DIR
    data_set = scale_dataset(
        load_data(data_dir=data_dir, without_duplicates=True, shuffle_data=False, rng=rng)
    )

    print(f"config = {config}")
    model_class = config["model"]["model_class"]

    if model_class == RandomBaseline.__name__:
        model = RandomBaseline(
            data=data_set,
            class_balance=config["model"]["class_balance"],
            rng=rng,
            random_seed=random_seed,
        )
    elif model_class == ZeroRate.__name__:
        model = ZeroRate(data=data_set, rng=rng, random_seed=random_seed)
    else:
        raise ValueError(f"Model class {model_class} does not exist.")

    # Predictions
    y_pred_val = model.predict(embeddings=data_set.X_val)

    eval_dict = evaluate(
        y_true=data_set.y_val,
        y_pred=y_pred_val,
        class_names_training=data_set.all_labels_train_sorted,
    )
    tune.report(**eval_dict)


def trial_dirname_creator(trial: trial.Trial) -> str:
    trial_dirname = f"{trial.trainable_name}_{trial.trial_id}_{str(trial.experiment_tag)}".replace(
        ": ", ", "
    )

    # max length for path in windosw = 260 character
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
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "random_seed": RANDOM_SEED,
            "model": tune.grid_search(
                [
                    {
                        "model_class": RandomBaseline.__name__,
                        "class_balance": tune.choice([True, False]),
                    },
                    {"model_class": ZeroRate.__name__},
                ]
            ),
        },
        progress_reporter=reporter,
    )

    print("Best config: ", analysis.get_best_config(metric="accuracy_avg", mode="max"))


if __name__ == "__main__":
    main()
