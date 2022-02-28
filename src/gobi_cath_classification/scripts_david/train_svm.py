# Import basic functionalities
import ray
import torch
from ray import tune

# Import functions located in directory packages
from gobi_cath_classification.pipeline import torch_utils
from gobi_cath_classification.pipeline.train_eval import training_function
from gobi_cath_classification.scripts_david.models import SupportVectorMachine


def main():
    ########################################################################################
    # FUNCTION NAME     : main()
    # INPUT PARAMETERS  : none
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Main function to test the SVM model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 18.02.2022
    # UPDATE            : 20.02.2022 - Model not longer forced on the GPU
    #                                  Addition of hyper parameters
    ########################################################################################

    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    # Print the current device the code is running on
    device = torch_utils.get_device()
    print(f"device = {device}")
    # Check if graphic precessing unit is available on the machine
    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 1}
    else:
        resources_per_trial = {"cpu": 1}

    # Configure a tune.CLIReporter as reporter to gain information during program flow
    reporter = tune.CLIReporter(
        max_report_frequency=10,
        infer_limit=10,
    )
    # Initialize ray
    ray.init()
    # Run machine learning model using tune.run and catch statistics in analysis
    analysis = tune.run(
        training_function,
        resources_per_trial=resources_per_trial,
        num_samples=1,
        config={
            "random_seed": 0,  # Random Seeds have no application for SVMs
            "class_weights": "inverse",  # No weighting of any kind implemented yet
            "model": {
                "model_class": SupportVectorMachine.__name__,
                "num_epochs": tune.choice([1]),
                "gamma": tune.choice([0.1]),
                "regularization": tune.choice(
                    [0.1]
                ),  # High regularization leads to large increase in computing time
                "kernel_function": tune.choice(["linear"]),
                "degree": tune.choice([0]),
                # After testing 144 different hyper parameter configurations and combinations, the following
                # composition was considered best {'model_class': 'SupportVectorMachine', 'num_epochs': 1,
                # 'gamma': 0.1, 'regularization': 0.1, 'kernel_function': 'linear', 'degree': 0}
            },
        },
        progress_reporter=reporter,
    )
    # Print the best configuration from analysis
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    # Run main function ...
    main()
