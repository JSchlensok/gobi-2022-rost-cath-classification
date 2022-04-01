import datetime as dt
import logging
from argparse import ArgumentParser
from typing import List, Any

import pytorch_lightning as pl
import pytorch_metric_learning as pml
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest.bayesopt import BayesOptSearch

from gobi_cath_classification.contrastive_learning_arcface import ArcFaceModel
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir
from gobi_cath_classification.pipeline.data.lightning import DataModule

ROOT_DIR = get_base_dir()
DATA_DIR = ROOT_DIR / "data"


# TODO set logging level as parameter
def tune_arcface(
        config,
        args,
        accuracy_calculator,
        num_epochs=50,
        data_dir=DATA_DIR,
        callbacks: List[Any]=[],

):

    data = DataModule(DATA_DIR, config["batch_size"])
    data.setup()

    tune_callback = TuneReportCheckpointCallback(
        {
            "val_loss": "val_loss",
            "val_acc_top10": "val_acc_top10"
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

def main(args, num_samples=10, num_epochs=20, data_dir=DATA_DIR):
    logging.basicConfig(filename="debug.log", filemode="a", level=logging.DEBUG)

    pl.seed_everything(42, workers=True)

    ##############
    # Tune setup #
    ##############

    config = {
        "layer_sizes": tune.choice([[512, 256], [512, 128], [512, 64], [256, 128], [256, 64], [128, 64]]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "loss_lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=num_epochs,
        grace_period=3,
    )

    reporter = tune.CLIReporter(
        metric_columns=["val_loss", "val_acc_top10", "training_iteration"],
        parameter_columns=["layer_sizes", "lr", "loss_lr", "batch_size"]
    )

    search_alg = BayesOptSearch(metric="val_loss", mode="min")

    ###############
    # Util setup #
    ###############


    accuracy_calculator = pml.utils.accuracy_calculator.AccuracyCalculator(include=["precision_at_1", "mean_average_precision"], k=10)

    timestamp = dt.datetime.today().astimezone(dt.timezone(dt.timedelta(hours=2))).strftime('%Y-%m-%d_%H%M')
    model_name = f"arcface_{timestamp}"


    progress_bar = (
        pl.callbacks.TQDMProgressBar()
    )  # TODO remove v_num https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#modifying-the-progress-bar

    # Run tune
    analysis = tune.run(
        tune.with_parameters(
            tune_arcface,
            args=args,
            accuracy_calculator=accuracy_calculator,
            num_epochs=num_epochs,
            data_dir=data_dir,
        ),
        resources_per_trial={
            "cpu": 1,
            "gpu": 1 if args.accelerator == "gpu" else 0
        },
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        name="tune" + model_name,
        local_dir=str(ROOT_DIR / "models"),
        verbose=0,
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

