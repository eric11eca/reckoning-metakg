<p align="center">
  <img src="./reckoning.png" alt="A robot is memorizing knowledge from books." title="reckoning-head" width=50% />
</p>

# RECKONING: Reasoning through Dynamic Knowledge Encoding


This codebase contains the implementation of the paper [RECKONING: Reasoning through Dynamic Knowledge Encoding](https://arxiv.org/abs/2305.06349) in proceedings of the [NeurIPS 2023](https://nips.cc/) conference.

## Quick links

* [Overview](#overview)
* [Requirements](#requirements)
* [Run experiments](#run-experiments)
  * [Quick start](#quick-start)
* [Bugs or questions](#bugs-or-questions)
* [Citation](#citation)


## Overview


RECKONING is a bi-level learning algorithm that teaches language models to reason by updating their parametric knowledge through back-propagation, allowing them to answer questions using the updated parameters. 

During training, the inner loop rapidly adapts a copy of the model weights to encode contextual knowledge into its parameters. In the outer loop, the model learns to use the updated weights to reproduce and answer reasoning questions about the memorized knowledge.

You can find more details of this work in our [paper](https://arxiv.org/abs/2305.06349).

## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```
To get the newest version of Higher, a meta-learning package, install from source through GitHub:
```
git clone git@github.com:facebookresearch/higher.git
cd higher
pip install .
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold independent to the versions.

## Run Experiments

### Quick start
Our experiments are managed by the framework [Hydra](https://github.com/facebookresearch/hydra)  and also [Weights&Bias](https://wandb.ai/).

No need to create new folders for recording the training results. Folders for each task will be automatically created to save checkpoints and logs. Then you can run our code with the following example:

```bash
python  run_maml.py  experiment=meta_train_gpt2
```

Users define the experiment name [meta_train_gpt2](https://github.com/eric11eca/reckoning-metakg/blob/main/config/experiment/meta_train_gpt2.yaml). It corresponds to the hydra configuration script for this experiment. The configuration files are stored in the `config` folder:
```
├── config
|	├── default
│   │   ├── default.yaml
│   ├── experiment
│   │   ├── meta_train_gpt2.yaml
│   │   ├── ...
│   ├── run.yaml
```
**Note**: You can overwrite the default arguments in the experiment YAML files.


## Bugs or questions
Note that this codebase is purely for the purpose of research and scientific experiments.  We expect unknown bugs or issues caused by different versions of updates in the past. If you encounter any problems when using the code or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker! If you have any questions related to the code or the paper, feel free to email [zeming.chen@epfl.ch](`zeming.chen@epfl.ch`). 

This repository will likely be outdated soon as we plan to move to a  new Meta-Learning package called [Betty](https://github.com/leopard-ai/betty/tree/main) for better scalability on the current LLM and distributed training requirements.

## Citation

Please cite our paper if you use RECKONING in your work:

```bibtex
@inproceedings{chen2023reckoning,
	title={{RECKONING}: Reasoning through Dynamic Knowledge Encoding},
	author={Zeming Chen and Gail Weiss and Eric Mitchell and Asli Celikyilmaz and Antoine Bosselut},
	booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	year={2023},
	url={https://openreview.net/forum?id=dUAcAtCuKk}
}
@misc{chen2023reckoning,
	title={RECKONING: Reasoning through Dynamic Knowledge Encoding}, 
	author={Zeming Chen and Gail Weiss and Eric Mitchell and Asli Celikyilmaz and Antoine Bosselut},
	year={2023},
	eprint={2305.06349},
	archivePrefix={arXiv},
	primaryClass={cs.CL}
}
```
