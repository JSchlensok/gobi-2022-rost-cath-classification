import ray
import torch
from ray import tune

from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.data import REPO_ROOT_DIR
from gobi_cath_classification.pipeline.train_eval import training_function


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
    # Check if GPU is available
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = torch_utils.get_device()
    print(f"device = {device}")

    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 1}
    else:
        resources_per_trial = {"cpu": 1}

    # Open CLI Reporter
    reporter = tune.CLIReporter(
        max_report_frequency=10,
        infer_limit=10,
    )

    # Where ever i save my ray results
    local_dir = "WHERE I WANT TO SAVE MY RAY RESULTS AND CHECKPOINTS"

    # Initialize Ray
    ray.init()
    # Start tune run
    analysis = tune.run(
        training_function,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "model": {
                # path of the folder with all saved files from local_dir onwards...
                "checkpoint_dir": tune.choice(
                    [
                        "training_function_2022-03-09_01-45-07\\training_function_2ad2b_00000_0_class_weights=inverse,layer_sizes=[1024],lr=1e-05,optimizer=adam,random_seed=1_2022-03-09_01-45-07"
                    ]
                ),
                "local_dir": tune.choice([local_dir]),
            }
        },
        progress_reporter=reporter,
        local_dir=local_dir,
    )
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    main()
