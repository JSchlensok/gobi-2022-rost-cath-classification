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
    )
    dataset.scale()

    device = torch_utils.get_device()
    print(f"device = {device}")
    path_dist_model = Path("/content/model_object.model")

    # model_path = Path("")

    print(f"\nmodel_path = {path_dist_model}")
    model = torch.load(path_dist_model, map_location=torch.device("cpu"))
    model.device = device

    y_pred_val = model.predict(embeddings=dataset.X_val)
    y_pred_test = model.predict(embeddings=dataset.X_test)

    eval_val = Evaluation(
        y_true=dataset.y_val,
        predictions=y_pred_val,
        train_labels=dataset.train_labels,
        model_name="NeuralNetworkModel",
    )

    eval_val.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
    eval_val_dict = {}
    for k, v in eval_val.eval_dict.items():
        eval_val_dict = {**eval_val_dict, **eval_val.eval_dict[k]}
    print(f"Evaluation VAL")
    for k, v in eval_val_dict.items():
        print(f"{k}: {v}")

    eval_test = Evaluation(
        y_true=dataset.y_test,
        predictions=y_pred_test,
        train_labels=dataset.train_labels,
        model_name="DistanceModel",
    )

    eval_test.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True)
    eval_test_dict = {}
    for k, v in eval_test.eval_dict.items():
        eval_test_dict = {**eval_test_dict, **eval_test.eval_dict[k]}
    print(f"\nEvaluation on TEST SET")
    eval_test.print_evaluation()


if __name__ == "__main__":
    main()
