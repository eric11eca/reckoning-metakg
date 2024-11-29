import os
import yaml
import torch
import torch.distributed
import higher
import logging
import lightning as pl

from pathlib import Path
from typing import Dict
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

from meta_kg.dataset import MetaKnowledgeDataset, create_dataloader
from meta_kg.model import CausalLM, MetaReasonSeq2Seq
from meta_kg.optimizer import LSLRSchedular
from meta_kg.inference import LLM_Generator
from meta_kg.evaluate import eval
from meta_kg.utils.py_io import write_generations, write_metrics

util_logger = logging.getLogger("meta_knowledge.module")

def reg_loss(parameters, reference_parameters, reg_lambda=0.2):
    loss = 0
    for p1, p2 in zip(parameters, reference_parameters):
        loss += torch.sum(torch.pow((p1 - p2), 2))

    return reg_lambda * loss

class MetaModule(pl.LightningModule):
    def __init__(self, config, logger):
        """Creates model runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__()
        self.model_logger = logger
        self.hparams.update(OmegaConf.to_container(config))
        self.global_trainin_step = 0
        self.global_epoch_counter = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_train_epoch_end(self):
        """Called at the end of the training epoch

        :param outputs: the outputs of the train step
        :rtype: None
        """
        self.global_epoch_counter += 1

    def validation_eval(self, results):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            gathered_args = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_args, self.hparams)
            self.hparams.update(gathered_args[0])
        metrics, _ = eval(results)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            os.makedirs(self.hparams.run_dir, exist_ok=True)
            os.makedirs(self.hparams.gen_dir, exist_ok=True)
            os.makedirs(self.hparams.eval_dir, exist_ok=True)

            config_name = os.path.join(self.hparams.run_dir, "config.yaml")
            if not os.path.exists(config_name):
                with open(config_name, "w") as f:
                    yaml.dump(self.hparams, f)
            log_name = f"outputs_{self.global_trainin_step}.jsonl"
            write_generations(results, self.hparams.gen_dir, log_name)
            log_name = f"eval_{self.global_trainin_step}.jsonl"
            write_metrics(metrics, self.hparams.eval_dir, log_name)
        elif not torch.distributed.is_initialized():
            os.makedirs(self.hparams.run_dir, exist_ok=True)
            os.makedirs(self.hparams.gen_dir, exist_ok=True)
            os.makedirs(self.hparams.eval_dir, exist_ok=True)

            config_name = os.path.join(self.hparams.run_dir, "config.yaml")
            if not os.path.exists(config_name):
                with open(config_name, "w") as f:
                    yaml.dump(self.hparams, f)
            log_name = f"outputs_{self.global_trainin_step}.jsonl"
            write_generations(results, self.hparams.gen_dir, log_name)
            log_name = f"eval_{self.global_trainin_step}.jsonl"
            write_metrics(metrics, self.hparams.eval_dir, log_name)

        # artifact = wandb.Artifact(f"test_metrics", type="dataset")
        # artifact.add_file(output_dir)
        # wandb.run.log_artifact(artifact)
        return metrics

    def validation_epoch_logic(self, outputs):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch
        :param outputs: the outputs of the train step
        :rtype: None
        """
        if self.trainer.num_devices > 1:
            outputs = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(outputs, self.validation_step_outputs)
            outputs = [item for sublist in outputs for item in sublist]
        else:
            outputs = self.validation_step_outputs

        if len(outputs) == 0:
            if self.hparams.multi_task:
                self.log(f"val_acc_label", 0.50, on_epoch=True, prog_bar=False)
            else:
                self.log(f"val_acc", 0.50, on_epoch=True, prog_bar=False)
            return

        # if self.trainer.is_global_zero:
        val_loss, generations = self.validation_epoch_logic(outputs)
        self.log("val_loss", val_loss.to(self.device), on_epoch=True, prog_bar=True, sync_dist=True)

        metrics = {}
        metrics = self.validation_eval(generations)
        for key in metrics:
            metrics[key] = round(metrics[key], 4)

        for key, val in metrics.items():
            self.log(f"val_{key}", torch.tensor(val).to(self.device), on_epoch=True, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> Dict:
        test_out = self.validation_step(batch, batch_idx)
        self.test_step_outputs.append(test_out)
        return test_out

    def test_epoch_logic(self, outputs):
        raise NotImplementedError

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        val_loss, generations = self.test_epoch_logic(outputs)
        self.log("test_loss", val_loss, on_epoch=True, prog_bar=True)

        metrics = {}
        metrics = self.validation_eval(generations)
        for key in metrics:
            metrics[key] = round(metrics[key], 4)

        self.validation_step_outputs.clear()
        for key, val in metrics.items():
            self.log(f"test_{key}", val, on_epoch=True, prog_bar=True)

    def get_lr_scheduler(self):
        """Sets up the optimizer learning rate scheduler"""
        num_devices = self.hparams.n_gpu if torch.cuda.is_available() else 1
        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.gradient_accumulation_steps
            * num_devices
        )

        total_steps = (
            len(self.train_dataloader().dataset) / effective_batch_size
        ) * self.hparams.num_train_epochs
        self.hparams.warmup_steps = (
            total_steps / effective_batch_size
        ) * self.hparams.warmup_proportion

        self.model_logger.info(
            "total_steps computed for scheduler: %s, warmup step: %s"
            % (total_steps, str(self.hparams.warmup_steps))
        )

        scheduler = get_cosine_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def load_dataset(self):
        """Loads the dataset"""
        self.model_logger.info("Loading dataset")
        self.train_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_path=self.hparams.data_dir,
            data_type="train",
            is_training=True,
        )
        self.dev_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_path=self.hparams.data_dir,
            data_type="test",
            is_training=False,
        )
        self.test_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_path=self.hparams.data_dir,
            data_type="test",
            is_training=False,
        )
        self.model_logger.info("Dataset loaded")

    def train_dataloader(self):
        """Loader to building training data.

        :rtype: DataLoader
        """
        dataloader = create_dataloader(self.hparams, self.train_data, is_training=True)
        self.model_logger.info("Length of training data loader %d" % len(dataloader))
        return dataloader

    def val_dataloader(self):
        """Loader to building validation data.

        :rtype: DataLoader
        """
        dataloader = create_dataloader(self.hparams, self.test_data, is_training=False)
        self.model_logger.info("Length of validation data loader %d" % len(dataloader))
        return dataloader

    def test_dataloader(self):
        """Loader to building test data.

        :rtype: DataLoader
        """
        dataloader = create_dataloader(self.hparams, self.test_data, is_training=False)
        self.model_logger.info("Length of test data loader %d" % len(dataloader))
        return dataloader


class MetaLearnerModule(MetaModule):
    def __init__(self, config):
        super().__init__(config, util_logger)
        util_logger.info("Running KG-MAML model")
        if config.model_type == "t5":
            self.model = MetaReasonSeq2Seq.from_config(config)
        else:
            self.model = CausalLM.from_config(config)
        self.tokenizer = self.model.tokenizer
        self.load_dataset()
        self.inner_lr_schedular_config(config.n_inner_iter, config.inner_lr)
        self.model_logger.info(
            f"Loaded runner instance, global_epoch_counter={self.global_epoch_counter}"
        )

    def inner_lr_schedular_config(self, n_inner_iter, inner_lr):
        self.inner_schedular = LSLRSchedular(
            num_inner_iter=n_inner_iter, init_lr=inner_lr)
        params_opt = list(
            filter(lambda p: p[1].requires_grad, self.model.named_parameters()))
        self.inner_schedular.initialization(self.model.named_parameters(), params_opt)

    def inner_loop_step(self, features, fmodel, diffopt) -> Dict:
        """Runs a single inner loop step

        :param features: the target batch
        :param fmodel: the fast model
        :param diffopt: the optimizer
        :rtype: dict
        :returns: dictionary that includes loss
        """
        for iter in range(self.hparams.n_inner_iter):
            for batch in features:
                train_out = fmodel(batch)
                train_loss = train_out["loss"]
                train_loss /= self.hparams.inner_accumulate_steps
                if not train_loss.requires_grad:
                    train_loss.requires_grad = True

                # reg = reg_loss(
                #     fmodel.parameters(),
                #     self.outer.parameters())

                if self.hparams.dyna_lr:
                    named_params = self.model.named_parameters()
                    self.inner_schedular.step(diffopt, named_params, iter)
                diffopt.step(train_loss)

    def inner_loop_end(self, features, print_out, fmodel) -> Dict:
        """Runs a single inner loop step

        :param features: the target batch
        :param print_out: None in this step
        :param fmodel: the fast model
        :rtype: dict
        :returns: dictionary that includes loss
        """
        with torch.no_grad():
            loss_total = torch.tensor(0.0, device=self.device)
            for batch in features:
                train_pred = fmodel(batch)
                loss_total += train_pred["loss"]
            return {"loss": loss_total / len(features)}

    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single meta-training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """
        inner_opt = self.config_inner_optimizer()
        loss = torch.tensor(0.0, device=self.device)
        outer_loss = torch.tensor(0.0, device=self.device)
        inner_loss = torch.tensor(0.0, device=self.device)
        inner_loss_diff = torch.tensor(0.0, device="cpu")

        # for _, task in enumerate(batch):
        train_features, dev_features, print_out = get_features(
            batch, self.hparams.inner_accumulate_steps
        )

        higher_grads = is_train
        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=higher_grads,
            accumulation_steps=self.hparams.inner_accumulate_steps
        ) as (fmodel, diffopt):
            inner_track, inner_out_prev = {}, {}

            for _, param in fmodel.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True

            if self.hparams.inner_verbose:
                inner_out_prev = self.inner_loop_end(
                    train_features, print_out, fmodel
                )

            self.inner_loop_step(train_features, fmodel, diffopt)
            inner_out_post = self.inner_loop_end(train_features, print_out, fmodel)
            inner_loss += inner_out_post["loss"].detach()

            if self.hparams.inner_verbose:
                diff = self._inner_loss_difference(
                    inner_out_prev, inner_out_post, inner_track
                )
                inner_loss_diff += diff
                self._inner_token_loss(inner_out_prev, inner_out_post, inner_track)

            if is_train:
                dev_out = fmodel(dev_features)
                loss += dev_out["loss"]
                outer_loss += dev_out["loss"].detach()
                records = []
            else:
                with torch.no_grad():
                    dev_out = fmodel(dev_features)
                    outer_loss += dev_out["loss"].detach()

                    records = []
                    for prompt, response in zip(
                        print_out["prompt"],
                        print_out["response"]):
                        records.append({
                            "guid": print_out["guid"],
                            "prompt": prompt,
                            "answer": response
                        })
                    self.validation_inference(records, fmodel)
        output_dict = {
            "loss": loss,
            "inner_loss": inner_loss, #/ len(batch),
            "outer_loss": outer_loss #/ len(batch),
        }

        if self.hparams.inner_verbose:
            output_dict["inner_loss_diff"] = inner_loss_diff / len(batch)

        if not is_train:
            output_dict["records"] = records

        return output_dict

    def _inner_token_loss(self, inner_out_prev, inner_out_post, inner_track):
        token_loss = []
        prev_token_loss = inner_out_prev["token_loss"]
        post_token_loss = inner_out_post["token_loss"]
        for prev_loss, post_loss in zip(prev_token_loss, post_token_loss):
            for token in post_loss:
                curr = post_loss[token]
                prev = prev_loss[token]
                diff = curr - prev
                post_loss[token] = (curr, diff)
            token_loss.append(post_loss)
        inner_track["token_loss"] = [token_loss]

    def _inner_loss_difference(self, inner_out_prev, inner_out_post, inner_track):
        diff = inner_out_post["loss"].detach() - inner_out_prev["loss"].detach()
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
                f"batch_{mkey}",
                output_dict[mkey],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True
            )
        self.global_trainin_step += 1
        return output_dict

    def validation_inference(self, records, fmodel):
        meta_model = CausalLM.from_config(self.hparams)
        meta_model.load_state_dict(fmodel.state_dict())
        generator = LLM_Generator(
            model_repo_id="gpt2",
            model=meta_model.lm_model,
            tokenizer=fmodel.tokenizer,
            device=self.device)
        generator.generate(
            records,
            pad_token_id=fmodel.tokenizer.eos_token_id,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=256
        )

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
                f"val_batch_{mkey}",
                output_dict[mkey],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True
            )

        self.validation_step_outputs.append(output_dict)
        return output_dict

    def validation_epoch_logic(self, outputs):
        val_loss = torch.stack([x["outer_loss"].cpu() for x in outputs]).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["records"]
        return val_loss, print_out_flatten

    def test_epoch_logic(self, outputs):
        test_loss = torch.stack([x["outer_loss"].cpu() for x in outputs]).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["print_out"]
        return test_loss, print_out_flatten


class MetaLMModule(MetaLearnerModule):
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
                model_params.append({"params": param, "lr": self.hparams.inner_lr})
        inner_opt = torch.optim.AdamW(
            model_params, betas=(0.9, 0.95))
        return inner_opt

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p
            for n, p in self.model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p
            for n, p in self.model.named_parameters()
            if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {"params": parameters_first, "weight_decay": self.hparams.weight_decay},
            {"params": parameters_sec, "weight_decay": 0.0},
            {"params": self.inner_schedular.parameters(), "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            betas=(0.9, 0.95)
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

        if config.model_type == "t5":
            self.model = MetaReasonSeq2Seq.from_config(config)
        else:
            self.model = CausalLM.from_config(config)
        self.tokenizer = self.model.tokenizer

        self.load_dataset()
        self.model_logger.info(
            f"Loaded runner instance, global_epoch_counter={self.global_epoch_counter}"
        )

    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """
        features = {
            "input_ids": batch["input_ids"].to(torch.device(self.hparams.device)),
            "attention_mask": batch["attention_mask"].to(
                torch.device(self.hparams.device)),
            "labels": batch["labels"].to(torch.device(self.hparams.device)),
        }

        if is_train:
            out = self.model(features)
            output_dict = {
                "loss": out["loss"],
                "train_loss": out["loss"].cpu()
            }
        else:
            with torch.no_grad():
                out = self.model(features)
                output_dict = {
                    "loss": out["loss"].cpu(),
                    "print_out": batch["print_out"],
                }

        return output_dict

    def training_step(self, batch, batch_idx) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        output_dict = self.step(batch, is_train=True)
        self.log(
            "batch_train_loss",
            output_dict["train_loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.global_trainin_step += 1
        return output_dict

    def validation_inference(self, records, model):
        generator = LLM_Generator(
            model_repo_id="gpt2",
            model=model.lm_model,
            tokenizer=model.tokenizer,
            device=self.device)
        generator.generate(
            records,
            pad_token_id=model.tokenizer.eos_token_id,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=256
        )

    def validation_step(self, batch, batch_idx) -> Dict:
        """Runs a single validation step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        output_dict = self.step(batch, is_train=False)
        self.log(
            f"val_batch_loss",
            output_dict["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        records = []
        print_out = output_dict["print_out"]
        for prompt, response in zip(
            print_out["prompt"],
            print_out["response"]):
            records.append({
                "guid": print_out["guid"],
                "prompt": prompt,
                "answer": response
            })
        self.validation_inference(records, self.model)
        output_dict["records"] = records
        self.validation_step_outputs.append(output_dict)
        return output_dict

    def validation_epoch_logic(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["records"]
        return val_loss, print_out_flatten

    def test_epoch_logic(self, outputs):
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["records"]
        return test_loss, print_out_flatten

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p for n, p in self.model.named_parameters()
            if not any(nd in n for nd in no_decay)]
        parameters_sec = [
            p for n, p in self.model.named_parameters()
            if any(nd in n for nd in no_decay)]

        optimizer_grouped_parameters = [
            {"params": parameters_first, "weight_decay": self.hparams.weight_decay},
            {"params": parameters_sec, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


def batch_split(row):
    """Split a batch into multiple inner-loop batches

    :param row: a batch of data
    :rtype: list
    """
    inner_loader = DataLoader(row, batch_size=4, shuffle=False)
    splited = [inner_batch for inner_batch in inner_loader]
    return splited

def batch_aggregate(rb):
    """Aggregate a batch of data

    :param rb: a batch of data
    :rtype: dict
    """
    inputs, masks, types = rb[0], rb[1], rb[2]
    train_feature = {
        "input_ids": inputs,
        "attention_mask": masks,
        "token_type_ids": types,
        "evaluate": False,
    }
    return train_feature


def get_features(batch, accumulate_steps: int = 1):
    """Get features from batch

    :param batch: the target batch
    :param accumulate: whether to accumulate gradient
    :rtype: dict
    """
    print_out = batch["print_out"]
    train_features = {
        "input_ids": batch["train_input_ids"],
        "attention_mask": batch["train_attention_mask"],
        "labels": batch["train_labels"]
    }

    dev_features = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"]
    }

    global_batch_size = batch["train_input_ids"].shape[0]

    # print("global_batch_size: ", global_batch_size)
    # print("accumulate_steps: ", accumulate_steps)
    micro_batch_size = global_batch_size // accumulate_steps
    # print("micro_batch_size: ", micro_batch_size)


    micro_input_ids = batch["train_input_ids"].split(micro_batch_size)
    micro_attention_mask = batch["train_attention_mask"].split(micro_batch_size)
    micro_labels = batch["train_labels"].split(micro_batch_size)
    micro_batches = []

    for input_ids, attention_mask, labels in zip(
        micro_input_ids,
        micro_attention_mask,
        micro_labels):
        micro_batches.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })
    train_features = micro_batches

    # rebatch = [batch_split(batch[key]) for key in data_keys]
    # train_features = [batch_aggregate(rb) for rb in zip(*rebatch)]

    return train_features, dev_features, print_out
