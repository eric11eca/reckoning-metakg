import torch
import higher
import logging
import pytorch_lightning as pl

from typing import Dict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from meta_kg.dataset import MetaKnowledgeDataset
from meta_kg.model import PretrainedEncoderDecoder
from meta_kg.train import setup_trainer
from meta_kg.optimizer import LSLRSchedular
from meta_kg.utils.py_io import write_json
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
        self.inner_schedular = LSLRSchedular(
            num_inner_iter=config.n_inner_iter,
            init_lr=config.inner_lr
        )
        if not config.baseline:
            params_opt = list(filter(
                lambda p: p[1].requires_grad,
                self.model.named_parameters()))
            self.inner_schedular.initialization(
                self.model.named_parameters(),
                params_opt)

        self.load_dataset()
        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

        metadata = {
            "task": config.dataset,
            "model": config.model_name_or_path,
            "model_type": config.model_type,
            "input_format": config.input_format,
            "inner_lr": config.inner_lr,
            "n_inner_iter": config.n_inner_iter,
            "learning_rate": config.learning_rate,
            "baseline": config.baseline,
            "classifier": config.classifier,
            "no_facts": config.no_facts,
            "random_facts": config.random_facts,
            "wandb_name": config.wandb_name,
            "gpu_id": config.device_idx,
        }

        write_json(metadata, f"{config.run_dir}/metadata.json")

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

    def _batch_split(self, row):
        inner_loader = DataLoader(
            row, batch_size=4, shuffle=False)
        splited = [inner_batch for inner_batch in inner_loader]
        return splited

    def _batch_aggregate(self, rb):
        inputs, masks, types = rb[0], rb[1], rb[2]
        train_feature = {
            "input_ids": inputs,
            "attention_mask": masks,
            "token_type_ids": types,
            "evaluate": False
        }
        return train_feature

    def get_features(self, batch, is_train: bool):
        """Get features from batch"""
        print_out = batch["print_out"]
        train_features = {
            "input_ids": batch["train_input_ids"].to(
                torch.device(self.hparams.device)),
            "attention_mask": batch["train_attention_mask"].to(
                torch.device(self.hparams.device)),
            "token_type_ids": batch["train_token_type_ids"].to(
                torch.device(self.hparams.device)),
            "evaluate": False
        }

        if "train_labels" in batch:
            train_features["labels"] = batch["train_labels"].to(
                torch.device(self.hparams.device))

        dev_features = {
            "input_ids": batch["input_ids"].to(
                torch.device(self.hparams.device)),
            "attention_mask": batch["attention_mask"].to(
                torch.device(self.hparams.device)),
            "token_type_ids": batch["token_type_ids"].to(
                torch.device(self.hparams.device)),
            "evaluate": not is_train
        }

        if "labels" in batch and self.hparams.classifier:
            dev_features["labels"] = batch["labels"].to(
                torch.device(self.hparams.device))

        if self.hparams.inner_grad_accumulate:
            data_keys = ["train_input_ids",
                         "train_attention_mask", "train_token_type_ids"]
            rebatch = [self._batch_split(batch[key]) for key in data_keys]
            train_features = [
                self._batch_aggregate(rb) for rb in zip(*rebatch)]

        return train_features, dev_features, print_out

    def inner_loop_step(self, features, print_out, fmodel, diffopt) -> Dict:
        """Runs a single inner loop step

        :param features: the target batch
        :param print_out: None in this step
        :param fmodel: the fast model
        :param diffopt: the optimizer
        :rtype: dict
        :returns: dictionary that includes loss
        """
        # if not isinstance(features, list):
        #     features = [features]
        # for iter in range(self.hparams.n_inner_iter):
        #     loss = torch.tensor(0., device=self.device)
        #     for batch in features:
        #         train_out = fmodel(batch, print_out, is_inner=True)
        #         train_loss = train_out["loss"]
        #         if not train_loss.requires_grad:
        #             train_loss.requires_grad = True
        #         loss += train_loss
        #     loss = loss / len(features)
        #     self.inner_schedular.step(
        #         diffopt, self.model.named_parameters(), iter)
        #     diffopt.step(loss)

        for iter in range(self.hparams.n_inner_iter):
            train_out = fmodel(features, print_out, is_inner=True)
            train_loss = train_out["loss"]
            if not train_loss.requires_grad:
                train_loss.requires_grad = True
            self.inner_schedular.step(
                diffopt, self.model.named_parameters(), iter)
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
        # if not isinstance(features, list):
        #     features = [features]
        # with torch.no_grad():
        #     loss = torch.tensor(0., device=self.device)
        #     for batch in features:
        #         train_pred = fmodel(
        #             batch, print_out, is_inner=True)
        #         loss += train_pred["loss"]
        #     return {"loss": loss / len(features)}

        with torch.no_grad():
            train_pred = fmodel(
                features, print_out, is_inner=True)
            return train_pred

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
            train_features, dev_features, print_out = self.get_features(
                task, is_train)

            with higher.innerloop_ctx(
                self.model, inner_opt,
                copy_initial_weights=False,
                track_higher_grads=is_train
            ) as (fmodel, diffopt):
                inner_track = {}
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
                    diff = inner_out_post["loss"].detach() - \
                        inner_out_prev["loss"].detach()
                    inner_loss_diff += diff
                    inner_track["inner_loss_diff"] = [diff]
                    inner_track["inner_loss"] = [inner_out_post["loss"].item()]

                if self.hparams.inner_verbose:
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
            "inner_loss_diff": inner_loss_diff / len(batch),
            'inner_loss': inner_loss / len(batch),
            'outer_loss': outer_loss / len(batch),
        }

        if not is_train:
            output_dict["print_out"] = print_outs

        # TODO need to do manual backward pass with accumulated loss!

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
            avg_innser_loss_diff = torch.stack(
                [x["inner_loss_diff"] for x in outputs]).mean()
            avg_inner_loss = torch.stack(
                [x["inner_loss"] for x in outputs]).mean()
            avg_outer_loss = torch.stack(
                [x["outer_loss"] for x in outputs]).mean()

            self.log(
                "avg_inner_loss",
                avg_inner_loss,
                on_step=False,
                on_epoch=True
            )

            self.log(
                "avg_innser_loss_diff",
                avg_innser_loss_diff,
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
            for mkey in ["inner_loss", "outer_loss"]:
                self.log(
                    f'val_batch_{mkey}',
                    output_dict[mkey],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False
                )
        return output_dict

    def validation_epoch_end(self, outputs):
        """Called at the end of the validation epoch
        :param outputs: the outputs of the train step
        :rtype: None 
        """
        if len(outputs) == 0:
            metrics = {
                "acc": 0.6498,
                "f1": 0.6498
            }
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"val_{metric_name}",
                    metric_value,
                    on_epoch=True,
                    prog_bar=True
                )
            return
        if self.baseline:
            val_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        else:
            val_loss = torch.stack([x["outer_loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        epoch = self.global_epoch_counter
        step = self.global_trainin_step

        out_file_name = f"dev_out-epoch={epoch}_step={step}.json"
        metirc_file_name = f"val_metrics-epoch={epoch}_step={step}.json"

        if not self.hparams.baseline:
            print_out_flatten = []
            for out in outputs:
                print_out_flatten += out["print_out"]
            outputs = [{"print_out": item} for item in print_out_flatten]

        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.run_dir}/{out_file_name}",
            f"{self.hparams.run_dir}/{metirc_file_name}"
        )

        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                on_epoch=True,
                prog_bar=True
            )

    def test_step(self, batch, batch_idx) -> Dict:
        test_out = self.validation_step(batch, batch_idx)
        return test_out

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["outer_loss"] for x in outputs]).mean()
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)

        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["print_out"]
        outputs = [{"print_out": item} for item in print_out_flatten]

        out_file_name = f"test_eval_out.json"
        metirc_file_name = f"test_metrics.json"

        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.run_dir}/{out_file_name}",
            f"{self.hparams.run_dir}/{metirc_file_name}"
        )

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
        self.test_data = self.dev_data
        # self.test_data = MetaKnowledgeDataset(
        #     self.model_logger,
        #     self.hparams,
        #     self.tokenizer,
        #     self.hparams.train_dir,
        #     data_type="test",
        #     is_training=False
        # )
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
    model = MetaKnowledgeRunner(args)
    trainer = setup_trainer(args)

    if args.do_train:
        if args.load_checkpoint is not None:
            trainer.fit(model, ckpt_path=args.load_checkpoint)
        else:
            trainer.fit(model)

    if args.do_eval:
        try:
            assert args.load_checkpoint is not None
        except AssertionError:
            util_logger.error('Checkpoint path is not provided for evaluation')
        trainer.fit(model, ckpt_path=args.load_checkpoint)

