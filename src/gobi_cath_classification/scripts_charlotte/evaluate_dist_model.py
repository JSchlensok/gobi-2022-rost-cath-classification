"""
Script to evaluate distance model in colab.
"""

import torch
from pathlib import Path

import numpy as np

from gobi_cath_classification.pipeline.Evaluation import Evaluation
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
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

    path_dist_model = Path("/content/model_object.model")
    print(f"\nmodel_path = {path_dist_model}")
    model = torch.load(path_dist_model, map_location=torch.device("cpu"))
    model.device = device

    # split evaluation into 2 parts because of too much RAM needed otherwise
    split_at = 194
    y_pred_tmph_1 = model.predict(embeddings=dataset.X_tmp_holdout[:split_at])
    y_pred_tmph_2 = model.predict(embeddings=dataset.X_tmp_holdout[split_at:])

    print(f"\nEvaluation on TMPH SET [:{split_at}]")
    eval_tmph_1 = Evaluation(
        y_true=dataset.y_tmp_holdout[:split_at],
        predictions=y_pred_tmph_1,
        train_labels=dataset.train_labels,
        model_name="DistanceModel",
    )

    eval_tmph_1.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
    eval_tmph_1.compute_std_err(bootstrap_n=bootstrap_n)
    eval_tmph_1.print_evaluation()

    print(f"\nEvaluation on TMPH SET [{split_at}:]")
    eval_tmph_2 = Evaluation(
        y_true=dataset.y_tmp_holdout[split_at:],
        predictions=y_pred_tmph_2,
        train_labels=dataset.train_labels,
        model_name="DistanceModel",
    )

    eval_tmph_2.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
    eval_tmph_2.compute_std_err(bootstrap_n=bootstrap_n)
    eval_tmph_2.print_evaluation()


if __name__ == "__main__":
    main()
