import torch
import higher
import logging
import numpy as np
import pytorch_lightning as pl

from typing import Dict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from peft import (
    get_peft_model,
    PrefixTuningConfig,
    LoraConfig,
    TaskType
)

from meta_kg.dataset import MetaKnowledgeDataset
from meta_kg.model import MetaReasonLM
from meta_kg.optimizer import LSLRSchedular
from meta_kg.utils.py_io import write_json

util_logger = logging.getLogger(
    'meta_knowledge.module'
)


class MetaModule(pl.LightningModule):
    def __init__(self, config, logger):
        """Creates model runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__()
        self.model_logger = logger
        self.hparams.update(vars(config))

        self.global_trainin_step = 0
        self.global_epoch_counter = 0

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        metadata = {
            "task": config.dataset,
            "model": config.model_name_or_path,
            "model_type": config.model_type,
            "input_format": config.input_format,
            "inner_lr": config.inner_lr,
            "n_inner_iter": config.n_inner_iter,
            "learning_rate": config.learning_rate,
            "baseline": config.baseline,
            "no_facts": config.no_facts,
            "random_facts": config.random_facts,
            "wandb_name": config.wandb_name,
            "gpu_id": config.device_idx,
        }

        write_json(metadata, f"{config.run_dir}/metadata.json")

    def on_train_epoch_end(self):
        """Called at the end of the training epoch

        :param outputs: the outputs of the train step
        :rtype: None
        """
        self.global_epoch_counter += 1

    def validation_epoch_logic(self, outputs):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch
        :param outputs: the outputs of the train step
        :rtype: None
        """
        outputs = self.validation_step_outputs
        if len(outputs) == 0:
            if self.hparams.multi_task:
                self.log(f"val_acc_label", 0.50,
                         on_epoch=True, prog_bar=False)
            else:
                self.log(f"val_acc", 0.50,
                         on_epoch=True, prog_bar=False)
            return

        val_loss, outputs = self.validation_epoch_logic(outputs)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        epoch = self.global_epoch_counter
        step = self.global_trainin_step

        out_file_name = f"dev_out-epoch={epoch}_step={step}.json"
        metirc_file_name = f"val_metrics-epoch={epoch}_step={step}.json"

        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.run_dir}/{out_file_name}",
            f"{self.hparams.run_dir}/{metirc_file_name}"
        )

        self.validation_step_outputs.clear()

        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                on_epoch=True,
                prog_bar=True
            )

    def test_step(self, batch, batch_idx) -> Dict:
        test_out = self.validation_step(batch, batch_idx)
        self.test_step_outputs.append(test_out)
        return test_out

    def test_epoch_logic(self, outputs):
        raise NotImplementedError

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        test_loss, outputs = self.test_epoch_logic(outputs)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)

        out_file_name = f"test_eval_out.json"
        metirc_file_name = f"test_metrics.json"

        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.run_dir}/{out_file_name}",
            f"{self.hparams.run_dir}/{metirc_file_name}"
        )

        self.test_step_outputs.clear()
        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                on_epoch=True,
                prog_bar=True
            )

    def get_lr_scheduler(self):
        """Sets up the optimizer learning rate scheduler

        """
        num_devices = self.hparams.n_gpu if torch.cuda.is_available() else 1
        effective_batch_size = self.hparams.train_batch_size * \
            self.hparams.gradient_accumulation_steps * num_devices

        total_steps = (len(self.train_dataloader().dataset) /
                       effective_batch_size) * self.hparams.num_train_epochs
        self.hparams.warmup_steps = (
            total_steps / effective_batch_size
        ) * self.hparams.warmup_proportion

        self.model_logger.info(
            'total_steps computed for scheduler: %s, warmup step: %s' % (
                total_steps, str(self.hparams.warmup_steps))
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return scheduler

    def load_dataset(self):
        """Loads the dataset

        """
        self.model_logger.info('Loading dataset')
        self.train_data = MetaKnowledgeDataset(
            self.model_logger,
            self.hparams,
            self.tokenizer,
            self.hparams.train_dir,
            data_type="train",
            is_training=True
        )
        self.dev_data = MetaKnowledgeDataset(
            self.model_logger,
            self.hparams,
            self.tokenizer,
            self.hparams.train_dir,
            data_type="test",
            is_training=False
        )
        self.test_data = MetaKnowledgeDataset(
            self.model_logger,
            self.hparams,
            self.tokenizer,
            self.hparams.train_dir,
            data_type="test",
            is_training=False
        )
        self.model_logger.info('Dataset loaded')

    def train_dataloader(self):
        """Loader to building training data.

        :rtype: DataLoader
        """
        dataloader = self.train_data.load_dataloader()
        self.model_logger.info(
            'Length of training data loader %d' % len(dataloader)
        )
        return dataloader

    def val_dataloader(self):
        """Loader to building validation data.

        :rtype: DataLoader
        """
        dataloader = self.dev_data.load_dataloader()
        self.model_logger.info(
            'Length of validation data loader %d' % len(dataloader)
        )
        return dataloader

    def test_dataloader(self):
        """Loader to building test data.

        :rtype: DataLoader
        """
        dataloader = self.test_data.load_dataloader()
        self.model_logger.info(
            'Length of test data loader %d' % len(dataloader)
        )
        return dataloader


class MetaReasonLMModule(MetaModule):
    def __init__(self, config):
        super().__init__(config, util_logger)
        util_logger.info("Running KG-MAML model")

        self.model = MetaReasonLM.from_config(config)
        self.tokenizer = self.model.tokenizer
        self.load_dataset()
        self.inner_lr_schedular_config(
            config.n_inner_iter,
            config.inner_lr
        )
        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

    def inner_lr_schedular_config(self, n_inner_iter, inner_lr):
        self.inner_schedular = LSLRSchedular(
            num_inner_iter=n_inner_iter,
            init_lr=inner_lr)
        params_opt = list(filter(
            lambda p: p[1].requires_grad,
            self.model.named_parameters()))
        self.inner_schedular.initialization(
            self.model.named_parameters(),
            params_opt)

    def inner_loop_step(self, features, print_out, fmodel, diffopt) -> Dict:
        """Runs a single inner loop step

        :param features: the target batch
        :param print_out: None in this step
        :param fmodel: the fast model
        :param diffopt: the optimizer
        :rtype: dict
        :returns: dictionary that includes loss
        """
        for iter in range(self.hparams.n_inner_iter):
            try:
                train_out = fmodel(features, print_out, is_inner=True)
            except:
                print("inner loop error")
                print(print_out)
                continue
            train_loss = train_out["loss"]
            if not train_loss.requires_grad:
                train_loss.requires_grad = True
            if self.hparams.inner_mode == "all":
                named_params = self.model.named_parameters()
            else:
                named_params = self.trainable_params.items()
            self.inner_schedular.step(
                diffopt, named_params, iter)
            diffopt.step(train_loss)

    def inner_loop_end(self, features, print_out, fmodel, is_train: bool) -> Dict:
        """Runs a single inner loop step

        :param features: the target batch
        :param print_out: None in this step
        :param fmodel: the fast model
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """
        with torch.no_grad():
            try:
                train_pred = fmodel(
                    features, print_out, is_inner=True)
            except:
                print("inner loop error")
                print(print_out)
                return {"loss": torch.tensor(0., device=self.device)}
            return train_pred

    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single meta-training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """
        inner_opt = self.config_inner_optimizer()
        loss = torch.tensor(0., device=self.device)
        outer_loss = torch.tensor(0., device='cpu')
        inner_loss = torch.tensor(0., device='cpu')
        inner_loss_diff = torch.tensor(0., device='cpu')
        print_outs = []

        for _, task in enumerate(batch):
            train_features, dev_features, print_out = get_features(
                self.hparams.device,
                task,
                is_train,
                self.hparams.inner_grad_accumulate
            )

            with higher.innerloop_ctx(
                self.model, inner_opt,
                copy_initial_weights=False,
                track_higher_grads=is_train
            ) as (fmodel, diffopt):
                inner_track, inner_out_prev = {}, {}

                for _, param in fmodel.named_parameters():
                    if not param.requires_grad:
                        param.requires_grad = True

                if self.hparams.inner_verbose:
                    inner_out_prev = self.inner_loop_end(
                        train_features, print_out, fmodel, is_train)

                self.inner_loop_step(
                    train_features, print_out, fmodel, diffopt)
                inner_out_post = self.inner_loop_end(
                    train_features, print_out, fmodel, is_train)
                inner_loss += inner_out_post["loss"].detach().cpu()

                if self.hparams.inner_verbose:
                    diff = self._inner_loss_difference(
                        inner_out_prev, inner_out_post, inner_track)
                    inner_loss_diff += diff
                    self._inner_token_loss(
                        inner_out_prev, inner_out_post, inner_track)

                if is_train:
                    dev_out = fmodel(dev_features, print_out)
                    loss += dev_out["loss"]
                    outer_loss += dev_out["loss"].detach().cpu()
                else:
                    with torch.no_grad():
                        dev_out = fmodel(dev_features, print_out)
                        outer_loss += dev_out["loss"].detach().cpu()
                        dev_out["print_out"].update(inner_track)
                        print_outs.append(dev_out["print_out"])
        output_dict = {
            'loss': loss,
            'inner_loss': inner_loss.detach() / len(batch),
            'outer_loss': outer_loss.detach() / len(batch),
        }

        if self.hparams.inner_verbose:
            output_dict["inner_loss_diff"] = inner_loss_diff / len(batch)

        if not is_train:
            output_dict["print_out"] = print_outs

        return output_dict

    def _inner_token_loss(self, inner_out_prev, inner_out_post, inner_track):
        token_loss = []
        prev_token_loss = inner_out_prev["token_loss"]
        post_token_loss = inner_out_post["token_loss"]
        for (prev_loss, post_loss) in zip(prev_token_loss, post_token_loss):
            for token in post_loss:
                curr = post_loss[token]
                prev = prev_loss[token]
                diff = curr - prev
                post_loss[token] = (curr, diff)
            token_loss.append(post_loss)
        inner_track["token_loss"] = [token_loss]

    def _inner_loss_difference(self, inner_out_prev, inner_out_post, inner_track):
        diff = inner_out_post["loss"].detach() - \
            inner_out_prev["loss"].detach()
        inner_track["inner_loss_diff"] = [diff]
        inner_track["inner_loss"] = [inner_out_post["loss"].item()]
        return diff

    def training_step(self, batch, batch_idx) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        output_dict = self.step(batch, is_train=True)
        for mkey in ["inner_loss", "outer_loss"]:
            self.log(
                f'batch_{mkey}',
                output_dict[mkey],
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )
        self.global_trainin_step += 1
        return output_dict

    def validation_step(self, batch, batch_idx) -> Dict:
        """Runs a single validation step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        torch.set_grad_enabled(True)
        self.model.train()
        output_dict = self.step(batch, is_train=False)
        for mkey in ["inner_loss", "outer_loss"]:
            self.log(
                f'val_batch_{mkey}',
                output_dict[mkey],
                on_step=True,
                on_epoch=False,
                prog_bar=False
            )
        self.validation_step_outputs.append(output_dict)
        return output_dict

    def validation_epoch_logic(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["print_out"]
        outputs = [{"print_out": item} for item in print_out_flatten]
        return val_loss, outputs

    def test_epoch_logic(self, outputs):
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["print_out"]
        outputs = [{"print_out": item} for item in print_out_flatten]
        return test_loss, outputs


class KGMAMLModule(MetaReasonLMModule):
    def __init__(self, config):
        super().__init__(config)

    def config_inner_optimizer(self):
        """Configures the inner loop optimizer

        :param model_params: the model parameters
        :rtype: torch.optim
        :returns: the optimizer
        """
        model_params = []
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                model_params.append(
                    {"params": param, "lr": self.hparams.inner_lr})

        if self.hparams.inner_opt == "adam":
            inner_opt = torch.optim.AdamW(
                model_params,
                amsgrad=False
            )
        else:
            inner_opt = torch.optim.SGD(
                model_params,
                momentum=0.9,
            )
        return inner_opt

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {
                "params": parameters_first,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": parameters_sec,
                "weight_decay": 0.0
            },
            {
                "params": self.inner_schedular.parameters(),
                "weight_decay": 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


class KGMAMLPrefixModule(MetaReasonLMModule):
    def __init__(self, config):
        super().__init__(config)

        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=config.prefix_dim
        )
        self.model.model = get_peft_model(
            self.model.model, peft_config)

        self.prefix_params = {}
        self.model_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prefix_params[name] = param
            else:
                self.model_params[name] = param

        self.lm_head = self.model.model.base_model.lm_head
        for name, param in self.lm_head.named_parameters():
            param.requires_grad = True
            self.prefix_params[name] = param

        self.inner_lr_schedular_config(
            config.n_inner_iter,
            config.inner_lr
        )

        self.num_prefix_params = len(self.prefix_params)
        self.num_model_params = len(self.model_params)

        util_logger.info(
            f"Number of prefix parameters: {self.num_prefix_params}")
        util_logger.info(
            f"Number of model parameters: {self.num_model_params}")

    def set_model_params_grad(self, grad: bool = True):
        for _, param in self.model_params.items():
            param.requires_grad = grad

    def set_prefix_params_grad(self, grad: bool = True):
        for _, param in self.prefix_params.items():
            param.requires_grad = grad

    def config_inner_optimizer(self):
        """Configures the inner loop optimizer

        :param model_params: the model parameters
        :rtype: torch.optim
        :returns: the optimizer
        """
        model_params = []
        for _, param in self.prefix_params.items():
            model_params.append(
                {"params": param, "lr": self.hparams.inner_lr})

        inner_opt = torch.optim.AdamW(
            model_params,
            amsgrad=False
        )
        return inner_opt

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """

        parameters_prefix = [
            p for _, p in self.prefix_params.items()
        ]

        optimizer_grouped_parameters = [
            {
                "params": parameters_prefix,
                "weight_decay": self.hparams.weight_decay
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


class KGMAMLLoraModule(MetaReasonLMModule):
    def __init__(self, config):
        super().__init__(config)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False, r=16,
            lora_alpha=32, lora_dropout=0.1
        )
        self.model.model = get_peft_model(
            self.model.model, peft_config)

        self.trainable_params = {}
        self.frozen_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.trainable_params[name] = param
            else:
                self.frozen_params[name] = param

        for name, param in self.model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
                self.trainable_params[name] = param

        for name, param in self.model.named_parameters():
            if np.any([x in name for x in ['.34.', '.35.']]):
                print(f"Enable training: {name}")
                param.requires_grad = True
                self.trainable_params[name] = param

        self.inner_lr_schedular_config(
            config.n_inner_iter,
            config.inner_lr
        )

    def set_model_params_grad(self, grad: bool = True):
        for _, param in self.model_params.items():
            param.requires_grad = grad

    def set_prefix_params_grad(self, grad: bool = True):
        for _, param in self.lora_params.items():
            param.requires_grad = grad

    def config_inner_optimizer(self):
        """Configures the inner loop optimizer

        :param model_params: the model parameters
        :rtype: torch.optim
        :returns: the optimizer
        """
        model_params = []
        for _, param in self.trainable_params.items():
            model_params.append(
                {"params": param, "lr": self.hparams.inner_lr})

        inner_opt = torch.optim.AdamW(
            model_params,
            amsgrad=False
        )
        return inner_opt

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p for n, p in self.trainable_params.items() if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p for n, p in self.trainable_params.items() if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {
                "params": parameters_first,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": parameters_sec,
                "weight_decay": 0.0
            },
            {
                "params": self.inner_schedular.parameters(),
                "weight_decay": 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


class CausalLMModule(MetaModule):
    def __init__(self, config):
        """Creates model runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__(config, util_logger)

        util_logger.info("Running baseline model")

        self.model = MetaReasonLM.from_config(config)
        self.tokenizer = self.model.tokenizer

        self.load_dataset()
        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """

        print_out = batch["print_out"]

        features = {
            "input_ids": batch["input_ids"].to(
                torch.device(self.hparams.device)),
            "attention_mask": batch["attention_mask"].to(
                torch.device(self.hparams.device)),
            "evaluate": not is_train
        }

        if "token_type_ids" in batch:
            features["token_type_ids"] = batch["token_type_ids"].to(
                torch.device(self.hparams.device))

        if "labels" in batch:
            features["labels"] = batch["labels"].to(
                torch.device(self.hparams.device))

        if is_train:
            out = self.model(features, print_out)
            loss = out["loss"]
            output_dict = {
                'loss': loss,
                'train_loss': loss.cpu()
            }
        else:
            with torch.no_grad():
                out = self.model(features, print_out)
                loss = out["loss"]
                output_dict = {
                    'loss': loss,
                    'train_loss': loss.cpu(),
                    'print_out': out["print_out"],
                }

        return output_dict

    def training_step(self, batch, batch_idx) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        output_dict = self.base_step(batch, is_train=True)
        self.log(
            f'batch_train_loss',
            output_dict["train_loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )
        self.global_trainin_step += 1
        return output_dict

    def validation_step(self, batch, batch_idx) -> Dict:
        """Runs a single validation step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        output_dict = self.base_step(batch, is_train=False)
        assert len(output_dict["print_out"]["gen_out"]) == len(
            output_dict["print_out"]["answer"])
        self.log(
            f'val_batch_loss',
            output_dict["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=False
        )
        self.validation_step_outputs.append(output_dict)
        return output_dict

    def validation_epoch_logic(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return val_loss, outputs

    def test_epoch_logic(self, outputs):
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return test_loss, outputs

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {
                "params": parameters_first,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": parameters_sec,
                "weight_decay": 0.0
            },
            {
                "params": self.inner_schedular.parameters(),
                "weight_decay": 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


def batch_split(row):
    inner_loader = DataLoader(
        row, batch_size=4, shuffle=False)
    splited = [inner_batch for inner_batch in inner_loader]
    return splited


def batch_aggregate(rb):
    inputs, masks, types = rb[0], rb[1], rb[2]
    train_feature = {
        "input_ids": inputs,
        "attention_mask": masks,
        "token_type_ids": types,
        "evaluate": False
    }
    return train_feature


def get_features(device, batch, is_train: bool, accumulate: bool):
    """Get features from batch"""
    print_out = batch["print_out"]
    train_features = {
        "input_ids": batch["train_input_ids"].to(
            torch.device(device)),
        "attention_mask": batch["train_attention_mask"].to(
            torch.device(device)),
        "token_type_ids": batch["train_token_type_ids"].to(
            torch.device(device)),
        "evaluate": False
    }

    if "train_labels" in batch:
        train_features["labels"] = batch["train_labels"].to(
            torch.device(device))

    dev_features = {
        "input_ids": batch["input_ids"].to(
            torch.device(device)),
        "attention_mask": batch["attention_mask"].to(
            torch.device(device)),
        "token_type_ids": batch["token_type_ids"].to(
            torch.device(device)),
        "evaluate": not is_train
    }

    if accumulate:
        data_keys = ["train_input_ids",
                     "train_attention_mask", "train_token_type_ids"]
        rebatch = [batch_split(batch[key]) for key in data_keys]
        train_features = [
            batch_aggregate(rb) for rb in zip(*rebatch)]

    return train_features, dev_features, print_out
