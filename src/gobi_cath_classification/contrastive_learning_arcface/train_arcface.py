import logging
from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
import pytorch_metric_learning as pml

from gobi_cath_classification.contrastive_learning_arcface import ArcFaceModel
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir
from gobi_cath_classification.pipeline.data.lightning import DataModule

ROOT_DIR = get_base_dir()
DATA_DIR = ROOT_DIR / "data"

# TODO set logging level as parameter
def main(args):
    logging.basicConfig(filename="debug.log", filemode="a", level=logging.DEBUG)

    pl.seed_everything(42, workers=True)

    data = DataModule(DATA_DIR, 64)
    data.setup()

    accuracy_calculator = pml.utils.accuracy_calculator.AccuracyCalculator()

    model_name = f"arcface_{datetime.today().strftime('%Y-%m-%d %H%M')}"

    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=model_name)

    progress_bar = (
        pl.callbacks.ProgressBar()
    )  # TODO remove v_num https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#modifying-the-progress-bar

    checkpointer = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=ROOT_DIR / f"models/{model_name}",
        filename="{epoch:02d}--{val_loss:.2f}",
        every_n_epochs=25,
    )

    trainer = pl.Trainer(
        benchmark=True,  # should lead to a speedup
        min_epochs=500,
        max_epochs=500,
        auto_lr_find=True,
        accelerator=args.accelerator,
        devices=1,
        precision=16 if args.accelerator == "gpu" else 32,
        callbacks=[progress_bar, checkpointer],
        logger=logger,
        num_sanity_val_steps=0,  # a training step runs in < 1 min, no need for this - will throw an error since the validation step that employs the lookup embeddings computed in ArcFaceModel.training_epoch_end() won't have them available
    )

    # TODO set hyperparameters
    model = ArcFaceModel(
        (data.num_classes, data.train_dataloader()),
        data.label_encoder,
        accuracy_calculator,
        model_name,
        1e-2,
        args.accelerator,
    )

    """
    lr_finder = trainer.tuner.lr_find(model, data.train_dataloader(), data.val_dataloader())
    fig = lr_finder.plot(suggest=True)
    fig.show()
    print(lr_finder.suggestion)
    print(type(lr_finder.suggestion()))
    """

    trainer.fit(model, data)


if __name__ == "__main__":
    # TODO parse hyperparameters
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu")
    args = parser.parse_args()

    main(args)
