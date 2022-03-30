import datetime as dt
import logging
from argparse import ArgumentParser
from typing import List, Any

import pytorch_lightning as pl
import pytorch_metric_learning as pml
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from gobi_cath_classification.contrastive_learning_arcface import ArcFaceModel
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir
from gobi_cath_classification.pipeline.data.lightning import DataModule

ROOT_DIR = get_base_dir()
DATA_DIR = ROOT_DIR / "data"


# TODO set logging level as parameter
def tune_arcface(
        config,
        accuracy_calculator,
        num_epochs=50,
        data_dir=DATA_DIR,
        callbacks: List[Any]=[],

):

    data = DataModule(DATA_DIR, config["batch_size"])
    data.setup()

    tune_callback = TuneReportCheckpointCallback(
        {
            "loss": "val_loss",
            "mean_accuracy": "val_acc_top10"
        },
        filename="checkpoint",
        on="validation_end"
    )

    logger = pl.loggers.TensorBoardLogger(tune.get_trial_dir(), name="", version=".")

    trainer = pl.Trainer(
        benchmark=True,  # should lead to a speedup
        max_epochs=num_epochs,
        accelerator=args.accelerator,
        devices=1,
        precision=16 if args.accelerator == "gpu" else 32,
        callbacks=[*callbacks, tune_callback],
        logger=logger,
        num_sanity_val_steps=0,  # a training step runs in < 1 min, no need for this - will throw an error since the validation step that employs the lookup embeddings computed in ArcFaceModel.training_epoch_end() won't have them available
        fast_dev_run=True if args.debug else False,
        #resume_from_checkpoint=str(ROOT_DIR / "models/checkpoint")
    )

    model = ArcFaceModel(
        config,
        (data.num_classes, data.train_dataloader()),
        data.label_encoder,
        accuracy_calculator,
        #model_name,
        args.accelerator,
    )

    trainer.fit(model, data)

def main(args, num_samples=10, num_epochs=50, data_dir=DATA_DIR):
    logging.basicConfig(filename="debug.log", filemode="a", level=logging.DEBUG)

    pl.seed_everything(42, workers=True)

    ##############
    # Tune setup #
    ##############

    config = {
        "l1_size": tune.choice([512, 256, 128]),
        "l2_size": tune.choice([256, 128, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "loss_lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    scheduler = tune.schedulers.PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-1),
            "loss_lr": tune.loguniform(1e-6, 1e-2),
            "batch_size": [32, 64, 128]
        }
    )

    reporter = tune.CLIReporter(
        parameter_columns=["l1_size", "l2_size", "lr", "loss_lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    ###############
    # Util setup #
    ###############


    accuracy_calculator = pml.utils.accuracy_calculator.AccuracyCalculator(include=["precision_at_1", "mean_average_precision"], k=10)

    timestamp = dt.datetime.today().astimezone(dt.timezone(dt.timedelta(hours=2))).strftime('%Y-%m-%d_%H%M')
    model_name = f"arcface_{timestamp}"


    progress_bar = (
        pl.callbacks.TQDMProgressBar()
    )  # TODO remove v_num https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#modifying-the-progress-bar

    checkpointer = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=ROOT_DIR / f"models/{model_name}",
        filename="{epoch:02d}--{val_loss:.2f}",
        every_n_epochs=25,
    )

    # Run tune
    analysis = tune.run(
        tune.with_parameters(
            tune_arcface,
            accuracy_calculator=accuracy_calculator,
            num_epochs=num_epochs,
            data_dir=data_dir,
            callbacks=[progress_bar]),
        resources_per_trial={
            "cpu": 1,
            "gpu": 1 if args.accelerator == "gpu" else 0
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_arcface",
        #local_dir=str(ROOT_DIR / "models"),
        #sync_config=tune.SyncConfig(syncer=None),
        #checkpoint_score_attr="mean_accuracy",
        #keep_checkpoints_num=1,
        #resume=True
    )

    print("Best hyperparameters: ", analysis.best_config)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)

