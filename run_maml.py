import logging

from meta_kg.train import setup_trainer
from meta_kg.utils.wandb_utils import setup_wandb

from meta_kg.module import (
    CausalLMModule,
    MetaReasonLMModule,
    MetaReasonPrefixLMModule
)

util_logger = logging.getLogger(
    'meta_knowledge.runner'
)


def get_module(args, module="all"):
    if module == "all":
        return init_kg_maml_module(args)
    elif module == "prefix":
        return init_kg_maml_prefix_module(args)
    else:
        return init_baseline_module(args)


def init_baseline_module(args):
    return CausalLMModule(args)


def init_kg_maml_module(args):
    return MetaReasonLMModule(args)


def init_kg_maml_prefix_module(args):
    return MetaReasonPrefixLMModule(args)


def run(args):
    util_logger.info('Setting up configuration for model runner...')
    setup_wandb(args)

    if args.baseline:
        model = get_module(args, module="baseline")
    else:
        model = get_module(args, module=args.inner_mode)

    trainer = setup_trainer(args)

    if args.do_train:
        if args.load_checkpoint is not None:
            model = CausalLMModule.load_from_checkpoint(
                args.load_checkpoint,
                config=args
            )
        trainer.fit(model)

    if args.do_eval:
        try:
            assert args.load_checkpoint is not None
        except AssertionError:
            util_logger.error('Checkpoint path is not provided for evaluation')
        if args.baseline:
            trainer.test(model, ckpt_path=args.load_checkpoint)
        else:
            # train on 1 datapoint for gradient enabling in validation
            model = CausalLMModule.load_from_checkpoint(
                args.load_checkpoint,
                config=args
            )
            trainer.fit(model)
