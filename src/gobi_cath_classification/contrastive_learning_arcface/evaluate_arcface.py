import logging
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pickle
import pytorch_lightning as pl
import pytorch_metric_learning as pml

from gobi_cath_classification.contrastive_learning_arcface import ArcFaceModel
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir
from gobi_cath_classification.pipeline.data.lightning import DataModule
from gobi_cath_classification.pipeline.data import load_data

ROOT_DIR = get_base_dir()
DATA_DIR = ROOT_DIR / "data"


def main(args):
    print("Loading model ...")
    with open(ROOT_DIR / f"models/{args.model_name}.pickle", "rb") as f:
        model = pickle.load(f)

    print("Loading data ...")
    dataset = load_data(
        DATA_DIR,
        rng=np.random.RandomState(42),
        without_duplicates=True,
        shuffle_data=False,
        load_only_small_sample=False,
        reloading_allowed=True,
        encode_labels=True,
    )

    print("Making predictions ...")
    test_pred = model.predict(dataset.get_split("test", "embedding-tensor", False, "tensor")[0])

    pred_path = ROOT_DIR / ("predictions/" + f"{args.model_name}_test219.csv")
    print(f"Saving predictions to {pred_path} ...")
    test_pred.save(pred_path)


if __name__ == "__main__":
    # TODO create reusable config class
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    main(args)
