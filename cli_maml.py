from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
    parser.add_argument("--dataset", default="proofwriter_owa_natlang")
    parser.add_argument("--model_name_or_path",
                        default="gpt2", required=False)
    parser.add_argument("--model_type",
                        default="gpt2", required=False)

    parser.add_argument("--output_dir", default="output",
                        type=str, required=False)
    parser.add_argument("--expand_dev", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')

    # Meta Learn parameters
    parser.add_argument('--inner_lr', type=float, default=5e-5,
                        help='Inner loop learning rate for SGD')
    parser.add_argument("--n_inner_iter", default=5, type=int,
                        help="Total number of inner training epochs to perform.")

    # Model parameters
    parser.add_argument("--checkpoint", type=str)

    # Preprocessing/decoding-related parameters
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--max_output_length", default=16, type=int)

    # Training-related parameters
    parser.add_argument("--fact_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=1, type=int,
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
    parser.add_argument('--device', type=str, default='cuda:0')

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

    parser.add_argument("--wandb_api_key", type=str, default="5d22b1d85f1fd5bb0c5758b93903c364ee5dc93d",
                        help="The particular wandb api key to use [default='']")
    parser.add_argument('--wandb_entity', type=str, default='causal_scaffold')
    parser.add_argument('--wandb_project', type=str, default='meta_knowledge')
    parser.add_argument('--wandb_name', type=str,
                        default='gpt2_question_kg_proofwriter_owa_natlang')
    parser.add_argument('--wandb_data', type=str,
                        default='')
    parser.add_argument("--wandb_note",
                        dest="wandb_note",
                        default='empty',
                        type=str,
                        help="The note to use for the wandb [default='empty']")
    parser.add_argument("--wandb_model",
                        dest="wandb_model",
                        default='',
                        type=str,
                        help="Specifies a location to an existing wandb model [default='']")

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

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_dir:
            raise ValueError(
                "If `do_train` is True, then `train_dir` must be specified.")
        if not args.predict_dir:
            raise ValueError(
                "If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_predict:
        if not args.predict_dir:
            raise ValueError(
                "If `do_predict` is True, then `predict_dir` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))
    run(args)


if __name__ == '__main__':
    main()
