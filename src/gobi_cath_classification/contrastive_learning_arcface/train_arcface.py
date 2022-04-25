from argparse import ArgumentParser
import datetime as dt
import logging
import os

import numpy as np
import pickle
import pytorch_lightning as pl
import pytorch_metric_learning as pml

from gobi_cath_classification.pipeline import load_data
from gobi_cath_classification.contrastive_learning_arcface import ArcFaceModel
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir
from gobi_cath_classification.pipeline.data.lightning import DataModule

ROOT_DIR = get_base_dir()
DATA_DIR = ROOT_DIR / "data"

# TODO set logging level as parameter
def main(args):
    logging.basicConfig(filename="debug.log", filemode="a", level=logging.DEBUG)

    pl.seed_everything(42, workers=True)

    data = DataModule(DATA_DIR, args.batch_size)
    data.setup()

    timestamp = (
        dt.datetime.today().astimezone(dt.timezone(dt.timedelta(hours=2))).strftime("%Y-%m-%d_%H%M")
    )
    model_name = f"arcface_{timestamp}"

    logger = pl.loggers.TensorBoardLogger(ROOT_DIR / "lightning_logs", name=model_name)

    progress_bar = (
        pl.callbacks.ProgressBar()
    )  # TODO remove v_num https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#modifying-the-progress-bar

    checkpointer = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        dirpath=ROOT_DIR / f"models/{model_name}",
        filename="{epoch:02d}--{val_acc:.2f}",
        every_n_epochs=1,
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", min_delta=0.0, patience=10
    )

    trainer = pl.Trainer(
        benchmark=True,  # should lead to a speedup
        max_epochs=10 if args.debug else 500,
        accelerator=args.accelerator,
        devices=1,
        precision=16 if args.accelerator == "gpu" else 32,
        callbacks=[progress_bar, checkpointer, early_stopping],
        logger=logger,
        num_sanity_val_steps=0,  # a training step runs in < 1 min, no need for this - will throw an error since the validation step that employs the lookup embeddings computed in ArcFaceModel.training_epoch_end() won't have them available
        limit_train_batches=0.01 if args.debug else None,
    )

    model = ArcFaceModel(
        {
            "model_lr": args.lr,
            "loss_lr": args.loss_lr,
            "layer_sizes": [512, 128],
            "batch_size": args.batch_size,
            "pickle_intermediates": args.pickle_intermediates,
        },
        (data.num_classes, data.train_dataloader()),
        data.label_encoder,
        model_name,
        args.accelerator,
        args.subcenters,
    )

    trainer.fit(model, data)

    results = trainer.test(model, data.test_dataloader())

    print(results)

    if args.pickle_final:
        path = ROOT_DIR / f"models/{model_name}_final.pickle"
        print(f"Pickling to {path} ...")
        with open(path, "wb+") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    # TODO create reusable config class
    parser = ArgumentParser()

    # Hardware
    parser.add_argument("--accelerator", type=str, default="gpu")

    # Trainer
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--pickle-final", action="store_true")
    parser.add_argument("--pickle-intermediates", action="store-true")

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--loss-lr", type=float, default=0.075)
    parser.add_argument("--subcenters", action="store_true")
    parser.add_argument("--layer-sizes", type=int, nargs=2, default=[512, 128])

    args = parser.parse_args()

    main(args)
