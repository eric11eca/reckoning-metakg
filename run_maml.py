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

MODULE_DICT = {
    "all": MetaReasonLMModule,
    "prefix": MetaReasonPrefixLMModule
}



def run(args):
    util_logger.info('Setting up configuration for model runner...')
    setup_wandb(args)

    module_class = MODULE_DICT.get(args.inner_mode, CausalLMModule)
    trainer = setup_trainer(args)

    if args.do_train:
        if args.load_checkpoint is not None:
            model = module_class.load_from_checkpoint(
                args.load_checkpoint,
                config=args
            )
        else:
            model = module_class(args)
        trainer.fit(model)

    if args.do_eval:
        try:
            assert args.load_checkpoint is not None
        except AssertionError:
            util_logger.error('Checkpoint path is not provided for evaluation')
        if args.baseline:
            model = CausalLMModule(args)
            trainer.test(model, ckpt_path=args.load_checkpoint)
        else:
            # train on 1 datapoint for gradient enabling in validation
            model = module_class.load_from_checkpoint(
                args.load_checkpoint,
                config=args
            )
            trainer.fit(model)
