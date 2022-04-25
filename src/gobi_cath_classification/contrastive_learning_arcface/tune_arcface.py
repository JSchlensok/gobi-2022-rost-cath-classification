import datetime as dt
import logging
from argparse import ArgumentParser
from typing import List, Any

import pytorch_lightning as pl
import pytorch_metric_learning as pml
import ray
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
    model_name: str,
    num_epochs=50,
    data_dir=DATA_DIR,
    callbacks: List[Any] = [],
):

    data = DataModule(DATA_DIR, config["batch_size"])
    data.setup()

    tune_callback = TuneReportCheckpointCallback(
        ["train_loss", "val_loss", "val_acc_top10", "val_acc"],
        filename="checkpoint",
        on="validation_end",
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
        overfit_batches=0.01 if args.debug else 0.0,
        # resume_from_checkpoint=str(ROOT_DIR / "models/checkpoint")
    )

    model = ArcFaceModel(
        config,
        (data.num_classes, data.train_dataloader()),
        data.label_encoder,
        model_name,
        args.accelerator,
        args.subcenters,
    )

    trainer.fit(model, data)


def main(args, num_samples=10, num_epochs=20, data_dir=DATA_DIR):
    logging.basicConfig(filename="debug.log", filemode="a", level=logging.DEBUG)

    pl.seed_everything(42, workers=True)

    ray.init(logging_level=logging.ERROR, log_to_driver=False)

    ##############
    # Tune setup #
    ##############

    config = {
        "layer_sizes": [512, 128],
        "model_lr": tune.loguniform(1e-5, 1e-2),
        "loss_lr": tune.loguniform(1e-1, 1e-4),
        "batch_size": tune.choice([32, 64, 128, 256]),
    }

    if args.scheduler == "pbt":
        scheduler = tune.schedulers.PopulationBasedTraining(
            time_attr="time_total_s",
            metric="val_loss",
            mode="min",
            perturbation_interval=120,
            hyperparam_mutations=config,
        )

    elif args.scheduler == "asha":
        scheduler = tune.schedulers.ASHAScheduler(
            max_t=num_epochs, metric="val_loss", mode="min", grace_period=5
        )
    else:
        pass

    reporter = tune.CLIReporter(
        max_report_frequency=30,
        metric_columns=["train_loss", "val_loss", "val_acc", "val_acc_top25", "training_iteration"],
        parameter_columns=["model_lr", "loss_lr", "batch_size"],
    )

    if args.scheduler == "asha":
        search_alg = BayesOptSearch(metric="val_loss", mode="min")

    ###############
    # Util setup #
    ###############

    timestamp = (
        dt.datetime.today().astimezone(dt.timezone(dt.timedelta(hours=2))).strftime("%Y-%m-%d_%H%M")
    )
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

    stopper = tune.stopper.ExperimentPlateauStopper(
        metric="val_loss", mode="min", top=10, patience=5
    )

    ##############
    # Run tuning #
    ##############

    print(f"Model name: tune_{args.scheduler}_{model_name}")

    analysis = tune.run(
        tune.with_parameters(
            tune_arcface,
            args=args,
            model_name=model_name,
            num_epochs=args.num_epochs,
            data_dir=data_dir,
        ),
        resources_per_trial={"cpu": 2 if args.colab else 1, "gpu": 1 if args.colab else 0},
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=1 if args.debug else args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg if scheduler == "asha" else None,
        progress_reporter=reporter,
        name=f"tune_{args.scheduler}_{model_name}",
        local_dir=str(ROOT_DIR / "ray_results"),
        verbose=2,
        reuse_actors=True,
        stop=stopper
        # sync_config=tune.SyncConfig(syncer=None),
        # checkpoint_score_attr="mean_accuracy",
        # keep_checkpoints_num=1,
        # resume=True
    )

    print("Best hyperparameters: ", analysis.best_config)


if __name__ == "__main__":
    parser = ArgumentParser()
    # Hardware
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--colab", action="store_true")

    # Utilities
    parser.add_argument("--debug", action="store_true")

    # Scheduler
    parser.add_argument("--scheduler", type=str, choices=["asha", "pbt", "pb2"], default="pbt")
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-epochs", type=int, default=10)

    # Model choice
    parser.add_argument("--subcenters", action="store_true")
    parser.add_argument("--layer-sizes", type=int, nargs=2, default=[512, 128])

    args = parser.parse_args()

    main(args)
