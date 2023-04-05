import os
import logging
import pytorch_lightning as pl

from pprint import pformat

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    GradientAccumulationScheduler
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from .utils.wandb_utils import init_wandb_logger

util_logger = logging.getLogger('meta_knowledge.trainer')


def setup_trainer(args) -> pl.Trainer:
    """Sets up the trainer and associated call backs from configuration 

    :param configuration: the target configuration 
    :rtype: a trainer instance 
    """
    # args = argparse.Namespace(**config.__dict__)
    mode = "max"
    util_logger.info('mode=%s via %s' % (mode, args.callback_monitor))

    if not os.path.isdir(args.output_dir):
        util_logger.info('making target directory: %s' % args.output_dir)
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
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop_callback = EarlyStopping(
        monitor=args.callback_monitor,
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode=mode
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
        accumulator
    ]

    # train parameters
    train_params = dict(
        accelerator='gpu',
        devices=[args.device_idx],
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        precision=32,
        callbacks=callbacks,
        num_sanity_val_steps=4,
        log_every_n_steps=5,
        # val_check_interval=0.5,
        profiler="simple",
        # strategy="deepspeed_stage_2_offload",
        # auto_lr_find=args.auto_lr_find,
    )

    if args.wandb_project:
        train_params['logger'] = init_wandb_logger(args)
        train_params['logger'].log_hyperparams(vars(args))

    util_logger.info(
        "\n===========\n"+pformat(train_params)+"\n==========="
    )

    trainer = pl.Trainer(**train_params)

    return trainer
