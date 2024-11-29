import os
import logging
import lightning as pl

from pprint import pformat

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    GradientAccumulationScheduler,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from meta_kg.utils.wandb_utils import init_wandb_logger

util_logger = logging.getLogger("meta_knowledge.trainer")


def setup_trainer(args) -> pl.Trainer:
    """Sets up the trainer and associated call backs from configuration

    :param configuration: the target configuration
    :rtype: a trainer instance
    """
    # args = argparse.Namespace(**config.__dict__)
    if "loss" in args.callback_monitor:
        mode = "min"
    else:
        mode = "max"
    util_logger.info("mode=%s via %s" % (mode, args.callback_monitor))

    if not os.path.isdir(args.output_dir):
        util_logger.info("making target directory: %s" % args.output_dir)
        os.mkdir(args.output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.run_dir,
        monitor=args.callback_monitor,
        mode=mode,
        save_top_k=1,
        verbose=True,
        auto_insert_metric_name=True,
    )

    accumulator = GradientAccumulationScheduler(scheduling={0: 2})
    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_callback = EarlyStopping(
        monitor=args.callback_monitor,
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode=mode,
    )

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="bold magenta",
            processing_speed="grey82",
            metrics="bold blue",
        )
    )

    callbacks = [
        lr_monitor,
        checkpoint_callback,
        early_stop_callback,
        progress_bar,
    ]

    # train parameters
    train_params = dict(
        accelerator="gpu",
        devices=1,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        precision="bf16-mixed",
        # precision=32,
        callbacks=callbacks,
        num_sanity_val_steps=4,
        log_every_n_steps=1,
        val_check_interval=0.1,
        # strategy="deepspeed_stage_2",
        reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    if args.wandb_project:
        train_params["logger"] = init_wandb_logger(args)
        train_params["logger"].log_hyperparams(args)

    util_logger.info("\n===========\n" + pformat(train_params) + "\n===========")

    trainer = pl.Trainer(**train_params)

    return trainer
