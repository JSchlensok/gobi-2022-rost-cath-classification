import ray
import torch
from ray import tune

from gobi_cath_classification.scripts_charlotte import torch_utils
from gobi_cath_classification.scripts_charlotte.models import (
    NeuralNetworkModel,
)
from gobi_cath_classification.scripts_charlotte.train_eval import training_function


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
                        "lr": tune.choice([1e-2]),
                        "batch_size": 32,
                        "optimizer": tune.choice(["adam", "sgd"]),
                        "layer_sizes": [1024],
                    },
                ]
            ),
        },
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
