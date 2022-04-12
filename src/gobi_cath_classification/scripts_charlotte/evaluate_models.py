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

    device = torch_utils.get_device()
    print(f"device = {device}")

    # paths_predictions = Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/reproduce_prottucker").glob(
    #     "**/*"
    # )

    bootstrap_n = 200
    # paths_predictions = Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models/predictions").glob("**/*")
    #
    # for pred_path in paths_predictions:
    #     if "lstm" in str(pred_path):
    #         print(f"pred_path = {pred_path}")
    #         which_set = ""
    #         y_pred = read_in_proba_predictions(filepath=pred_path)
    #         print(f"y_pred.probabilities.shape = {y_pred.probabilities.shape}")
    #
    #         print(f"len(np.sum(y_pred.probabilities.values, axis=1)) = {len(np.sum(y_pred.probabilities.values, axis=1))}")
    #         # y_pred_labels = read_in_label_predictions(filepath=pred_path, train_labels=dataset.train_labels)
    #         # filepath = pred_path.parent / (str(pred_path).split("/")[-1].replace("labels", "probabilities"))
    #         # save_predictions(pred=y_pred_labels, filepath=filepath)
    #
    #         if "val" in str(pred_path):
    #             y_true = dataset.y_val
    #             which_set = "VAL"
    #         elif "test" in str(pred_path):
    #             y_true = dataset.y_test
    #             which_set = "TEST"
    #         elif "tmph" in str(pred_path) or "holdout" in str(pred_path):
    #             y_true = dataset.y_tmp_holdout
    #             which_set = "TMPH"
    #
    #         print(f"len(y_true) = {len(y_true)}")
    #         print(f"len(y_pred.probabilities) = {len(y_pred.probabilities)}")
    #         eval_tmph = Evaluation(
    #             y_true=y_true,
    #             predictions=y_pred,
    #             train_labels=dataset.train_labels,
    #             model_name="LSTM",
    #         )
    #
    #         eval_tmph.compute_metrics(accuracy=True, mcc=True, f1=True, kappa=True, bacc=False)
    #         print(f"\nEvaluation on {which_set} SET")
    #         # eval_tmph.print_evaluation()
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

    paths_best_models = sorted(Path("/Users/x/Downloads/content4").glob("**/*model_object.model"))
    paths_best_models = sorted(
        Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models/final_choices").glob(
            "**/*model_object.model"
        )
    )
    paths_best_models = sorted(
        Path("/Users/x/Downloads/content_FCNN_1_hidden_layer").glob("**/*model_object.model")
    )
    paths_best_models = sorted(
        Path(
            "/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models/final_choices/content_FCNN_1_hidden_layer_2_BEST"
        ).glob("**/*model_object.model")
    )
    paths_best_models = sorted(
        Path("/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models").glob("**/*model_object.model")
    )

    paths_best_models = sorted(
        Path("/Users/x/Downloads/content_FCNN_1_hidden_layer_4").glob("**/*model_object.model")
    )
    paths_best_models = sorted(
        Path(
            "/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models/final_choices/content_FCNN_1_hidden_layer_2_BEST/gobi-2022-rost-cath-classification/ray_results/training_function_2022-04-03_12-53-23/training_function_0c2ef_00000_0_BEST"
        ).glob("**/*model_object.model")
    )
    paths_best_models = sorted(
        Path(
            "/Users/x/Desktop/bioinformatik/SEM_5/GoBi/best_models/final_choices/content_FCNN_1_hidden_layer_2_BEST/gobi-2022-rost-cath-classification/ray_results/training_function_2022-04-03_12-53-23/training_function_0c2ef_00000_0_BEST"
        ).glob("**/*model_object.model")
    )
    paths_best_models = sorted(
        Path(
            "/Users/x/Downloads/content_7"
        ).glob("**/*model_object.model")
    )

    for model_path in sorted(paths_best_models):

        # if "content_FCNN_1_hidden_layer_2_BEST/gobi-2022-rost-cath-classification/ray_results/training_function_2022-04-03_12-53-23/training_function_0c2ef_00000_0" in str(model_path):
        if "" in str(model_path):
            print(f"\nmodel_path = {model_path}")
            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.device = device
            model.model.eval()
            model_name = model.__class__.__name__
            print(f"model_name = {model_name}")

            # VAL SET
            y_pred_val = model.predict(embeddings=dataset.X_val)
            eval_val = Evaluation(
                y_true=dataset.y_val,
                predictions=y_pred_val,
                train_labels=dataset.train_labels,
                model_name=model_name,
            )

            eval_val.compute_metrics(accuracy=True)
            # eval_val.compute_std_err(bootstrap_n=bootstrap_n)
            print(f"\nEvaluation on VAL SET")
            eval_val.print_evaluation()

            # TEST SET
            y_pred_test = model.predict(embeddings=dataset.X_test)
            eval_test = Evaluation(
                y_true=dataset.y_test,
                predictions=y_pred_test,
                train_labels=dataset.train_labels,
                model_name=model_name,
            )

            eval_test.compute_metrics(accuracy=True)
            # eval_test.compute_std_err(bootstrap_n=bootstrap_n)
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

            eval_tmph.compute_metrics(accuracy=True)
            # eval_tmph.compute_std_err(bootstrap_n=bootstrap_n)
            print(f"\nEvaluation on TMPH SET")
            eval_tmph.print_evaluation()


if __name__ == "__main__":
    main()
