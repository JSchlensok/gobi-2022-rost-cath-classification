from collections import Counter

import torch
from pathlib import Path

import numpy as np

from gobi_cath_classification.pipeline.Evaluation import Evaluation
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.prediction import (
    read_in_label_predictions,
    read_in_proba_predictions,
    save_predictions,
)
from gobi_cath_classification.pipeline.utils import torch_utils


def main():
    dataset = load_data(
        data_dir=DATA_DIR,
        rng=np.random.RandomState(1),
        without_duplicates=True,
        shuffle_data=False,
        reloading_allowed=True,
        load_tmp_holdout_set=True,
    )
    dataset.scale()

    bootstrap_n = 1000
    device = torch_utils.get_device()
    print(f"device = {device}")

    paths_best_models = sorted(
        Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models").glob("**/*model_object.model")
    )

    for model_path in sorted(paths_best_models):
        filter_for = "FCNN"
        if filter_for in str(model_path):
            print(f"\nmodel_path = {model_path}")
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.device = device
            model.model.eval()
            model_name = model.__class__.__name__
            print(f"model_name = {model_name}")

            for which_set in ["VAL", "TEST", "TMPH"]:

                print(f"\nEvaluation on {which_set} SET")
                if which_set == "VAL":
                    y_true = dataset.y_val
                    y_pred = model.predict(embeddings=dataset.X_val)
                if which_set == "TEST":
                    y_true = dataset.y_test
                    y_pred = model.predict(embeddings=dataset.X_test)
                elif which_set == "TMPH":
                    y_true = dataset.y_tmp_holdout
                    y_pred = model.predict(embeddings=dataset.X_tmp_holdout)

                eval_val = Evaluation(
                    y_true=y_true,
                    predictions=y_pred,
                    train_labels=dataset.train_labels,
                    model_name=model_name,
                )

                eval_val.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
                eval_val.compute_std_err(bootstrap_n=bootstrap_n)
                eval_val.print_evaluation()


if __name__ == "__main__":
    main()
