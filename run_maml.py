import time
import torch
import higher
import logging
import pytorch_lightning as pl

from typing import Dict
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from meta_kg.dataset import MetaKnowledgeDataset
from meta_kg.model import PretrainedEncoderDecoder
from meta_kg.train import setup_trainer
from meta_kg.utils.wandb_utils import setup_wandb


util_logger = logging.getLogger(
    'meta_knowledge.runner'
)


class MetaKnowledgeRunner(pl.LightningModule):

    def __init__(self, config):
        """Creates model runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__()
        self.model_logger = util_logger
        self.hparams.update(vars(config))
        self.baseline = config.baseline

        if self.baseline:
            util_logger.info("Running baseline model")

        self.global_trainin_step = 0
        self.global_epoch_counter = 0

        self.model = PretrainedEncoderDecoder.from_config(config)
        self.tokenizer = self.model.tokenizer

        self.load_dataset()
        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

    def base_step(self, batch, is_train: bool) -> Dict:
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

        if "labels" in batch:
            features["labels"] = batch["labels"].to(
                torch.device(self.hparams.device))

        if is_train:
            out = self.model(features, print_out)
            loss = out["loss"]
            output_dict = {
                'loss': loss,
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


    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single meta-training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """
        inner_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.inner_lr
        )

        print_out = batch["print_out"]

        train_features = {
            "input_ids": batch["train_input_ids"].to(
                torch.device(self.hparams.device)),
            "attention_mask": batch["train_attention_mask"].to(
                torch.device(self.hparams.device)),
            "evaluate": False
        }

        if "train_labels" in batch:
            train_features["labels"] = batch["train_labels"].to(
                torch.device(self.hparams.device))

        with higher.innerloop_ctx(
            self.model, inner_opt,
            copy_initial_weights=False,
            track_higher_grads=is_train
        ) as (fmodel, diffopt):
            for _ in range(self.hparams.n_inner_iter):
                train_out = fmodel(train_features, print_out)
                train_loss = train_out["loss"]
                diffopt.step(train_loss)

            with torch.no_grad():
                if is_train:
                    train_pred = fmodel(train_features, print_out)
                    inner_train_loss = train_pred["loss"].cpu()
                else:
                    # train_features["evaluate"] = True
                    inner_print_out = batch["inner_print_out"]
                    train_pred = fmodel(train_features, inner_print_out)
                    inner_train_loss = train_pred["loss"].cpu()

            dev_features = {
                "input_ids": batch["input_ids"].to(
                    torch.device(self.hparams.device)),
                "attention_mask": batch["attention_mask"].to(
                    torch.device(self.hparams.device)),
                "evaluate": not is_train
            }

            if "labels" in batch:
                dev_features["labels"] = batch["labels"].to(
                    torch.device(self.hparams.device))

            if is_train:
                dev_out = fmodel(dev_features, print_out)
                outer_train_loss = dev_out["loss"]
                output_dict = {
                    'loss': outer_train_loss,
                    'inner_loss': inner_train_loss,
                    'outer_loss': outer_train_loss.cpu(),
                }

            else:
                with torch.no_grad():
                    dev_out = fmodel(dev_features, print_out)
                    dev_out["print_out"]["inner_loss"] = [inner_train_loss.item()]
                    outer_train_loss = dev_out["loss"]
                    output_dict = {
                        'loss': outer_train_loss,
                        'inner_loss': inner_train_loss,
                        'outer_loss': outer_train_loss.cpu(),
                        'print_out': dev_out["print_out"],
                    }

                    # output_dict["inner_print_out"] = train_pred["print_out"]

        return output_dict

    def training_step(self, batch, batch_idx) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        if self.baseline:
            output_dict = self.base_step(batch, is_train=True)
            self.log(
                f'batch_train_loss',
                output_dict["train_loss"],
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )
        else:
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

    def training_epoch_end(self, outputs):
        """Called at the end of the training epoch

        :param outputs: the outputs of the train step
        :rtype: None 
        """
        if self.baseline:
            avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
            self.log(
                "avg_train_loss",
                avg_train_loss,
                on_step=False,
                on_epoch=True
            )
        else:
            avg_inner_loss = torch.stack([x["inner_loss"] for x in outputs]).mean()
            avg_outer_loss = torch.stack([x["outer_loss"] for x in outputs]).mean()

            self.log(
                "avg_inner_loss",
                avg_inner_loss,
                on_step=False,
                on_epoch=True
            )

            self.log(
                "avg_outer_loss",
                avg_outer_loss,
                on_step=False,
                on_epoch=True
            )

        self.global_epoch_counter += 1

    def validation_step(self, batch, batch_idx) -> Dict:
        """Runs a single validation step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        if self.baseline:
            output_dict = self.base_step(batch, is_train=False)
            assert len(output_dict["print_out"]["gen_out"]) == len(
                output_dict["print_out"]["answer"])
        else:
            torch.set_grad_enabled(True)
            self.model.train()
            output_dict = self.step(batch, is_train=False)
            assert len(output_dict["print_out"]["gen_out"]) == len(
                output_dict["print_out"]["answer"])
        return output_dict

    def validation_epoch_end(self, outputs):
        """Called at the end of the validation epoch
        :param outputs: the outputs of the train step
        :rtype: None 
        """
        if self.baseline:
            val_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        else:
            val_loss = torch.stack([x["outer_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        epoch = self.global_epoch_counter
        step = self.global_trainin_step
        timestr = time.strftime("%Y%m%d-%H%M%S")

        out_file_name = f"dev_eval_out-epoch={epoch}_step={step}_{timestr}.json"
        metirc_file_name = f"val_metrics-epoch={epoch}_step={step}_{timestr}.json"

        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.output_dir}/{out_file_name}",
            f"{self.hparams.output_dir}/{metirc_file_name}")

        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                on_epoch=True,
                prog_bar=True
            )
    
    def test_step(self, batch, batch_idx) -> Dict:
        print(batch["print_out"])
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["outer_loss"] for x in outputs]).mean()
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)

        timestr = time.strftime("%Y%m%d-%H%M%S")

        out_file_name = f"test_eval_out_{timestr}.json"
        metirc_file_name = f"test_metrics_{timestr}.json"

        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.output_dir}/{out_file_name}",
            f"{self.hparams.output_dir}/{metirc_file_name}")

        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                on_epoch=True,
                prog_bar=True
            )

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
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

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
            data_type="dev",
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


def run(args):
    util_logger.info('Setting up configuration for model runner...')

    setup_wandb(args)
    wandb_runner = None

    metrics = {}
    model = MetaKnowledgeRunner(args)
    trainer = setup_trainer(args)

    if args.do_train:
        if args.load_checkpoint is not None:
            trainer.fit(model, ckpt_path=args.load_checkpoint)
        else:
            trainer.fit(model)

        if args.wandb_project:
            wandb_runner = trainer.logger.experiment

        for key, value in trainer.callback_metrics.items():
            if key in ["log", "progress_bar"] or "acc" not in key:
                continue
            try:
                metrics[key] = value.detach().item()
            except:
                pass
    
    if args.do_eval:
        if args.load_checkpoint is not None:
            trainer.test(model, ckpt_path=args.load_checkpoint)
        else:
            trainer.test(model)
