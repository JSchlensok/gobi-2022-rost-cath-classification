import ray
import torch
from ray import tune

from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.torch_utils import RANDOM_SEED
from gobi_cath_classification.scripts_charlotte.models import (
    NeuralNetworkModel,
)
from gobi_cath_classification.pipeline.train_eval import (
    training_function,
    resume_training
)

def main():
    ########################################################################################
    # FUNCTION NAME     : main()
    # INPUT PARAMETERS  : none
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : main function to coordinate resuming training of a model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 03.03.2022
    # UPDATE            : ---
    ########################################################################################
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
        resume_training,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "unique_ID": tune.choice(["b4bd828c-03e8-4ef2-827d-82b932364e78"]),
        },
        progress_reporter=reporter,
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
