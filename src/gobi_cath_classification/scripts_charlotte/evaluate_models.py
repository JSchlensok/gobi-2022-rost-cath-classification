from collections import Counter

import torch
from pathlib import Path

import numpy as np

from gobi_cath_classification.pipeline.Evaluation import Evaluation
from gobi_cath_classification.pipeline.data import load_data, DATA_DIR
from gobi_cath_classification.pipeline.prediction import read_in_label_predictions, \
    read_in_proba_predictions
from gobi_cath_classification.pipeline.utils import torch_utils


def main():
    dataset = load_data(
        data_dir=DATA_DIR,
        rng=np.random.RandomState(1),
        without_duplicates=True,
        shuffle_data=False,
        reloading_allowed=True,
        load_tmp_holdout_set=True
    )
    dataset.scale()

    device = torch_utils.get_device()
    print(f"device = {device}")

    # paths_predictions = Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/reproduce_prottucker").glob(
    #     "**/*"
    # )
    #
    bootstrap_n = 200
    #
    # for pred_path in paths_predictions:
    #     if "proba" in str(pred_path) and "tmph" in str(pred_path):
    #         print(f"pred_path = {pred_path}")
    #         which_set = ""
    #         y_pred = read_in_proba_predictions(filepath=pred_path)
    #         if "val" in str(pred_path):
    #             y_true = dataset.y_val
    #             which_set = "VAL"
    #         elif "test" in str(pred_path):
    #             y_true = dataset.y_test
    #             which_set = "TEST"
    #         elif "tmph" in str(pred_path):
    #             y_true = dataset.y_tmp_holdout
    #             which_set = "TMPH"
    #
    #         eval_tmph = Evaluation(
    #             y_true=y_true,
    #             predictions=y_pred,
    #             train_labels=dataset.train_labels,
    #             model_name="ProtTucker",
    #         )
    #
    #         eval_tmph.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True, bacc=False)
    #         print(f"\nEvaluation on {which_set} SET")
    #         eval_tmph.print_evaluation()
    #         if which_set == "TEST" or which_set == "TMPH":
    #             eval_tmph.compute_std_err(bootstrap_n=bootstrap_n)
    #         eval_tmph.print_evaluation()
    #
    #         eval_tmph = Evaluation(
    #             y_true=y_true,
    #             predictions=y_pred,
    #             train_labels=dataset.train_labels,
    #             model_name="ProtTucker",
    #         )
    #
    #         eval_tmph.compute_metrics(accuracy=False, mcc=False, f1=False, kappa=False, bacc=True)
    #         if which_set == "TEST":
    #             eval_tmph.compute_std_err(bootstrap_n=bootstrap_n)
    #         print(f"\nEvaluation on {which_set} SET")
    #         eval_tmph.print_evaluation()

    paths_best_models = sorted(
        Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models/final_choices").glob(
            "**/*model_object.model"
        )
    )

    paths_best_models = sorted(
        Path("/Users/x/Downloads/content_FCNN_1_hidden_layer").glob(
            "**/*model_object.model"
        )
    )
    for model_path in paths_best_models:

        print(f"\nmodel_path = {model_path}")
        if "FCNN" in str(model_path):
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.device = device
            model_name = model.__class__.__name__
            print(f"model_name = {model_name}")

            # TEST SET
            y_pred_test = model.predict(embeddings=dataset.X_test)
            eval_test = Evaluation(
                y_true=dataset.y_test,
                predictions=y_pred_test,
                train_labels=dataset.train_labels,
                model_name=model_name,
            )

            eval_test.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True, bacc=True)
            eval_test.compute_std_err(bootstrap_n=bootstrap_n)
            print(f"\nEvaluation on TEST SET")
            eval_test.print_evaluation()


            # TMPH SET
            y_pred_tmph = model.predict(embeddings=dataset.X_tmp_holdout)
            eval_tmph = Evaluation(
                y_true=dataset.y_tmp_holdout,
                predictions=y_pred_tmph,
                train_labels=dataset.train_labels,
                model_name=model_name,
            )

            eval_tmph.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True, bacc=True)
            eval_tmph.compute_std_err(bootstrap_n=bootstrap_n)
            print(f"\nEvaluation on TMPH SET")
            eval_tmph.print_evaluation()

            # y_pred_tmph = model.predict(embeddings=dataset.X_tmp_holdout)
            # eval_tmph = Evaluation(
            #     y_true=dataset.y_tmp_holdout,
            #     predictions=y_pred_tmph,
            #     train_labels=dataset.train_labels,
            #     model_name=model_name,
            # )
            #
            # eval_tmph.compute_metrics(accuracy=False, mcc=False, f1=False, kappa=False, bacc=True)
            # eval_tmph.compute_std_err(bootstrap_n=bootstrap_n)
            # print(f"\nEvaluation on TMPH SET")
            # eval_tmph.print_evaluation()


if __name__ == "__main__":
    main()
