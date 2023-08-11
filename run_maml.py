from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import hydra
import random
import numpy as np
import torch

from typing import Optional
from omegaconf import OmegaConf, DictConfig, open_dict

from meta_kg.runner import run


@hydra.main(version_base="1.3", config_path="./config", config_name="run.yaml")
def main(args: DictConfig) -> Optional[float]:
    config_dict = OmegaConf.to_container(args, resolve=True)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_filename)),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("meta_knowledge.run_maml")
    logger.info(config_dict)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open_dict(args):
        args.device = f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu"
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        if not args.train_dir:
            raise ValueError(
                "If `do_train` is True, then `train_dir` must be specified."
            )
        if not args.predict_dir:
            raise ValueError(
                "If `do_train` is True, then `predict_dir` must be specified."
            )

    if args.do_eval and not args.predict_dir:
        raise ValueError("If `do_eval` is True, then `predict_dir` must be specified.")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"{args.output_dir}/{timestr}"
    os.makedirs(run_dir, exist_ok=True)

    with open_dict(args):
        args.run_dir = run_dir

    args.wandb_name = f"{args.dataset.replace('owa_', '')}-{args.wandb_name}"
    run(args)


if __name__ == "__main__":
    main()
