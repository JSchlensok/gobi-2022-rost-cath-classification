import ray
import torch
from ray import tune

from gobi_cath_classification.pipeline.utils import torch_utils
from gobi_cath_classification.pipeline.data import REPO_ROOT_DIR
from gobi_cath_classification.scripts_charlotte.models import DistanceModel
from gobi_cath_classification.pipeline.train_eval import training_function


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
    local_dir = REPO_ROOT_DIR / "ray_results"

    ray.init()
    analysis = tune.run(
        training_function,
        local_dir=local_dir,
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
                "model_class": DistanceModel.__name__,
                "distance_order": tune.grid_search([2]),
            },
        },
        progress_reporter=reporter,
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
