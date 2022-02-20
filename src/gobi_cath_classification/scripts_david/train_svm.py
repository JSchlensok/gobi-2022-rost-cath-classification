# Import basic functionalities
import ray
import torch
from ray import tune
# Import functions located in directory packages
from src.gobi_cath_classification.pipeline import torch_utils
from src.gobi_cath_classification.pipeline.train_eval import training_function
from src.gobi_cath_classification.scripts_david.models_test import TestSupportVectorMachine
from src.gobi_cath_classification.scripts_david.models import SupportVectorMachine

def main():
    ########################################################################################
    # FUNCTION NAME     : main()
    # INPUT PARAMETERS  : none
    # OUTPUT PARAMETERS : none
    # DESCRIPTION       : Main function to test the SVM model
    # AUTHOR            : D. Mauder
    # CREATE DATE       : 18.02.2022
    # UPDATE            : 20.02.2022 - Zwang der Berechnung auf die GPU entfernt
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
            "random_seed": 0,               # Random Seeds have no application for SVMs
            "class_weights": "inverse",     # No weighting of any kind implemented yet
            "model": {
                "model_class": SupportVectorMachine.__name__,   # Model = SupportVectorMachine
                "num_epochs": 1,                                # SVMs are fitted by using one fitting
                "lr": None,                                     # Learning Rate has no application for SVMs
                "batch_size": None,                             # Batch size has no application for SVMs
                "optimizer": None,                              # Optimizer has no application for SVMs
                "layer_sizes": None,                            # Layer size has no application for SVMs
            },
        },
        progress_reporter=reporter,
    )
    # Print the best configuration from analysis
    print("Best config: ", analysis.get_best_config(metric="accuracy_h", mode="max"))


if __name__ == "__main__":
    # Run main function ...
    main()
