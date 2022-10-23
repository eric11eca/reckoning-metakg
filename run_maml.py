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
        self.global_epoch_counter = 0

        self.model = PretrainedEncoderDecoder.from_config(config)
        self.tokenizer = self.model.tokenizer

        self.load_dataset()

        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single meta-training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """

        train_features = {
            "input_ids": batch["train_input_ids"].to(
                torch.device("cuda:1")),
            "attention_mask": batch["train_attention_mask"].to(
                torch.device("cuda:1")),
            "labels": batch["train_labels"].to(
                torch.device("cuda:1")),
            "evaluate": batch["evaluate"]
        }

        train_loader = batch["train_loader"]

        dev_features = {
            "input_ids": batch["input_ids"].to(
                torch.device(self.hparams.device)),
            "attention_mask": batch["attention_mask"].to(
                torch.device(self.hparams.device)),
            "labels": batch["labels"].to(
                torch.device(self.hparams.device)),
            "evaluate": batch["evaluate"]
        }

        inner_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.inner_lr
        )

        print_out = batch["print_out"]

        with higher.innerloop_ctx(
            self.model, inner_opt,
            copy_initial_weights=False
        ) as (fmodel, diffopt):
            for _ in range(self.hparams.n_inner_iter):
                for batch in train_loader:
                    train_features = {
                        "input_ids": batch["input_ids"].to(
                            torch.device("cuda:1")),
                        "attention_mask": batch["attention_mask"].to(
                            torch.device("cuda:1")),
                        "labels": batch["input_ids"].to(
                            torch.device("cuda:1")),
                        "evaluate": False
                    }
                    train_out = fmodel(train_features, print_out)
                    train_loss = train_out["loss"]
                    diffopt.step(train_loss)

            with torch.no_grad():
                inner_train_loss = []
                for batch in train_loader:
                    train_features = {
                        "input_ids": batch["input_ids"].to(
                            torch.device("cuda:1")),
                        "attention_mask": batch["attention_mask"].to(
                            torch.device("cuda:1")),
                        "labels": batch["input_ids"].to(
                            torch.device("cuda:1")),
                        "evaluate": False
                    }
                    train_pred = fmodel(train_features, print_out)
                    inner_train_loss.append(train_pred["loss"].cpu())

            if is_train:
                dev_out = fmodel(dev_features, print_out)
                outer_train_loss = dev_out["loss"]
                output_dict = {
                    'loss': outer_train_loss,
                    'inner_loss': torch.mean(inner_train_loss),
                    'outer_loss': outer_train_loss.cpu(),
                }

            else:
                with torch.no_grad():
                    dev_out = self.model(dev_features, print_out)
                    outer_train_loss = dev_out["loss"]
                    output_dict = {
                        'loss': outer_train_loss,
                        'inner_loss': torch.mean(inner_train_loss),
                        'outer_loss': outer_train_loss.cpu(),
                        'print_out': dev_out["print_out"],
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
        for mkey in ["inner_loss", "outer_loss"]:
            self.log(
                f'batch_{mkey}',
                output_dict[mkey],
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

        return output_dict

    def training_epoch_end(self, outputs):
        """Called at the end of the training epoch

        :param outputs: the outputs of the train step
        :rtype: None 
        """
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
        return output_dict

    def validation_epoch_end(self, outputs):
        """Called at the end of the validation epoch
        :param outputs: the outputs of the train step
        :rtype: None 
        """
        val_loss = torch.stack([x["outer_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        metrics_out = self.model.evaluate_output(outputs)

        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"val_{metric_name}",
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

        # self.train_data.load_dataset(self.tokenizer)
        # self.dev_data.load_dataset(self.tokenizer)
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


def run(args):
    util_logger.info('Setting up configuration for model runner...')

    setup_wandb(args)
    wandb_runner = None

    metrics = {}
    model = MetaKnowledgeRunner(args)

    if args.do_train:
        trainer = setup_trainer(args)
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
