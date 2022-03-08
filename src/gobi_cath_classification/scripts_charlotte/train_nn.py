import ray
import torch
from ray import tune

from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.data import REPO_ROOT_DIR
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED
from gobi_cath_classification.scripts_charlotte.models import (
    NeuralNetworkModel,
    DistanceModel,
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
            "class_weights": tune.grid_search(
                [
                    "inverse",
                ]
            ),
            "model": {
                "model_class": NeuralNetworkModel.__name__,
                "num_epochs": 200,
                "lr": tune.grid_search([1e-5, 1e-4, 1e-3]),
                "batch_size": 32,
                "optimizer": tune.choice(["adam"]),
                "layer_sizes": tune.choice(
                    [
                        [1024],
                        [1024, 1024],
                    ]
                ),
            },
        },
        progress_reporter=reporter,
        local_dir=REPO_ROOT_DIR / "src" / "gobi_cath_classification" / "model checkpoints",
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
