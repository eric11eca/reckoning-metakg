import os
import wandb
import logging

from omegaconf import OmegaConf

from meta_kg.train import setup_trainer
from meta_kg.utils.wandb_utils import setup_wandb

from meta_kg.module import (
    CausalLMModule,
    MetaLMModule,
)

from meta_kg.module_peft import MetaLMLoraModule, CausalLoraModule

util_logger = logging.getLogger("meta_knowledge.runner")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    util_logger.info("Setting up configuration for model runner...")

    setup_wandb(args)
    wandb.config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    if args.baseline:
        args.callback_monitor = "val_acc"

    if args.baseline and not args.use_lora:
        module_class = CausalLMModule
        util_logger.info("Running baseline model")
    elif args.baseline and args.use_lora:
        module_class = CausalLoraModule
        util_logger.info("Running LoRA baseline model")
    elif not args.baseline and args.use_lora:
        module_class = MetaLMLoraModule
        util_logger.info("Running LoRA MAML model")
    else:
        module_class = MetaLMModule
        util_logger.info("Running MAML model")

    trainer = setup_trainer(args)

    if args.do_train:
        if args.load_checkpoint and args.checkpoint is not None:
            model = module_class.load_from_checkpoint(
                args.checkpoint, config=args, map_location=args.device
            )
        else:
            model = module_class(args)
        trainer.fit(model)

        if args.baseline:
            trainer.test(model)

    if args.do_eval:
        try:
            assert args.checkpoint is not None
        except AssertionError:
            util_logger.error("Checkpoint path is not provided for evaluation")

        if args.baseline:
            model = CausalLMModule(args)
            trainer.test(model, ckpt_path=args.checkpoint)
        else:
            # train on 1 datapoint for gradient enabling in validation
            model = module_class.load_from_checkpoint(
                args.checkpoint, config=args, map_location=args.device
            )
            trainer.fit(model)
