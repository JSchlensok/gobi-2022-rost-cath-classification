import ray
import torch
from ray import tune

from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.data import REPO_ROOT_DIR
from gobi_cath_classification.scripts_charlotte.models import (
    NeuralNetworkModel,
)
from gobi_cath_classification.pipeline.train_eval import (
    training_function,
)


def main():
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = torch_utils.get_device()
    print(f"device = {device}")

    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 1}
    else:
        resources_per_trial = {"cpu": 1}

    reporter = tune.CLIReporter(
        max_report_frequency=120,
        infer_limit=10,
    )
    local_dir = REPO_ROOT_DIR / "ray_results"

    ray.init()
    analysis = tune.run(
        training_function,
        local_dir=local_dir,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "random_seed": tune.grid_search([1]),
            "class_weights": "sqrt_inverse",
            "model": tune.grid_search(
                [


                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 800,
                        "lr": 1e-04,
                        "batch_size": 256,
                        "optimizer": "adam",
                        "loss_function": "CrossEntropyLoss",
                        "layer_sizes": [1024, 256],
                        "dropout_sizes": [0.003, None],
                        "scale": True,
                    },
                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 800,
                        "lr": 1e-04,
                        "batch_size": 256,
                        "optimizer": "adam",
                        "loss_function": "CrossEntropyLoss",
                        "layer_sizes": [1024, 512],
                        "dropout_sizes": [0.003, None],
                        "scale": True,
                    },
                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 800,
                        "lr": 1e-04,
                        "batch_size": 256,
                        "optimizer": "adam",
                        "loss_function": "CrossEntropyLoss",
                        "layer_sizes": [1024, 1024],
                        "dropout_sizes": [0.003, None],
                        "scale": True,
                    },
                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 800,
                        "lr": 1e-04,
                        "batch_size": 256,
                        "optimizer": "adam",
                        "loss_function": "CrossEntropyLoss",
                        "layer_sizes": [1024, 2048],
                        "dropout_sizes": [0.003, None],
                        "scale": True,
                    },
                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 800,
                        "lr": 1e-04,
                        "batch_size": 256,
                        "optimizer": "adam",
                        "loss_function": "CrossEntropyLoss",
                        "layer_sizes": [1024, 128],
                        "dropout_sizes": [0.003, None],
                        "scale": True,
                    },
                    {
                        "model_class": NeuralNetworkModel.__name__,
                        "num_epochs": 800,
                        "lr": 1e-04,
                        "batch_size": 256,
                        "optimizer": "adam",
                        "loss_function": "CrossEntropyLoss",
                        "layer_sizes": [1024, 128],
                        "dropout_sizes": [0.003],
                        "scale": True,
                    },

                ]
            ),
        },
        progress_reporter=reporter,
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
