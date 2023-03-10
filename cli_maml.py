from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import logging

import random
import numpy as np
import torch

from run_maml import run


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--train_dir", default="data")
    parser.add_argument("--predict_dir", default="data")
    parser.add_argument("--dataset", default="clutrr_6_hop")
    parser.add_argument("--dataset_type", default="clutrr")
    parser.add_argument("--model_name_or_path",
                        default="gpt2", required=False)
    parser.add_argument("--model_type",
                        default="gpt2", required=False)

    parser.add_argument("--output_dir", default="output",
                        type=str, required=False)
    parser.add_argument("--expand_dev", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--baseline", action='store_true')

    # Meta Learn parameters
    parser.add_argument('--inner_mode', type=str, default='open',
                        help='open book or closed book: [open, closed]')
    parser.add_argument('--inner_opt', type=str, default='sgd',
                        help='inner optimizer choice: [sgd, adam]')
    parser.add_argument('--inner_lr', type=float, default=1e-5,
                        help='Inner loop learning rate for SGD')
    parser.add_argument("--n_inner_iter", default=1, type=int,
                        help="Total number of inner training epochs to perform.")
    parser.add_argument("--inner_verbose", action='store_true', default=False,
                        help="Get detailed information and dynamic of inner loop learning.")
    parser.add_argument("--align", action='store_true', default=False,
                        help="Align the inner loop task with the outer loop task.")
    parser.add_argument("--multi_task", action='store_true', default=False,
                        help="Multi-task learning inner and outer tasks.")
    parser.add_argument("--multi_objective", action='store_true', default=False,
                        help="Multi-task learning inner and outer objective.")
    parser.add_argument("--fomaml", action='store_true', default=False,
                        help="Enable first order MAML.")
    parser.add_argument("--inner_grad_accumulate", action='store_true', default=False,
                        help="Enable inner loop gradient accumulation.")
    parser.add_argument("--inner_accumulate_steps", default=4, type=int,
                        help="Number of inner loop gradient accumulation steps.")

    # Preprocessing/decoding-related parameters
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--max_output_length", default=16, type=int)

    # Training-related parameters
    parser.add_argument("--fact_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.06, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--callback_monitor', type=str, default='val_acc')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--input_format', type=str, default='lm')
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--load_checkpoint', type=str,
                        default=None, help='path to checkpoint')
    parser.add_argument('--classifier', action='store_true', default=False,
                        help='whether to use the classifier mode')
    parser.add_argument('--no_facts', action='store_true', default=False,
                        help='whether to do sanity check with no facts')
    parser.add_argument('--random_facts', action='store_true', default=False,
                        help='whether to do sanity check with random facts')
    parser.add_argument('--load_order', type=str, default='in',
                        help='order to load the facts')
    parser.add_argument('--freeze_partial', action='store_true', default=False,
                        help='whether to freeze partial layers of the model')

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=100,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--max_data', type=int, default=0,
                        help="max number of data points to do prediction")

    parser.add_argument("--wandb_api_key", type=str, default="9edee5b624841e10c88fcf161d8dc54d4efbee29",
                        help="The particular wandb api key to use [default='']")
    parser.add_argument('--wandb_entity', type=str, default='epfl_nlp_phd')
    parser.add_argument('--wandb_project', type=str, default='meta-knowledge')
    parser.add_argument('--wandb_name', type=str,
                        default='baseline-clutrr_6_hop')
    parser.add_argument('--wandb_data', type=str, default='',
                        help="Specifies an existing wandb dataset")
    parser.add_argument("--wandb_model", type=str, default='',
                        help="Specifies an existing wandb model")
    parser.add_argument('--wandb_checkpoint', action='store_true', default=False,
                        help="Save best checkpoints to wandb")

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger("meta_knowledge.cli_maml")
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = 1
    args.device = f"cuda:{args.device_idx}" if torch.cuda.is_available(
    ) else "cpu"

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        if not args.train_dir:
            raise ValueError(
                "If `do_train` is True, then `train_dir` must be specified.")
        if not args.predict_dir:
            raise ValueError(
                "If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_eval and not args.predict_dir:
        raise ValueError(
            "If `do_eval` is True, then `predict_dir` must be specified.")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"{args.output_dir}/{timestr}"
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

    run(args)


if __name__ == '__main__':
    main()
