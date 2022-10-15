import torch
import higher

from typing import Dict
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup

from torch.optim import AdamW

from meta_kg.dataset import MetaKnowledgeDataset
from meta_kg.model import GeneratorModel

import torch

import pytorch_lightning as pl


class MetaKnowledgeRunner(pl.LightningModule, GeneratorModel):

    def __init__(self, config):
        """Creates model runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__()

        self.hparams.update(config)
        self.load_dataset()

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.hparams.model_name_or_path)
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))

        self.global_epoch_counter = 0
        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask, labels=labels)
        return outputs[0]

    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single meta-training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """

        train_features = {
            "train_input_ids": batch[0]["train_input_ids"].to(
                torch.device("cuda")),
            "train_attention_mask": batch[0]["train_attention_mask"].to(
                torch.device("cuda")),
            "train_labels": batch[0]["train_labels"].to(
                torch.device("cuda"))
        }

        dev_features = {
            "dev_input_ids": batch[0]["dev_input_ids"].to(
                torch.device("cuda")),
            "dev_attention_mask": batch[0]["dev_attention_mask"].to(
                torch.device("cuda")),
            "dev_labels": batch[0]["dev_labels"].to(
                torch.device("cuda"))
        }

        inner_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.inner_lr
        )

        with higher.innerloop_ctx(
            self.model, inner_opt,
            copy_initial_weights=False
        ) as (fmodel, diffopt):
            for _ in range(self.hparams.n_inner_iter):
                train_out = fmodel(train_features)
                train_loss = train_out["loss"]
                diffopt.step(train_loss)

            with torch.no_grad():
                train_pred = fmodel(train_features)
                inner_train_loss = train_pred["loss"].cpu()

            if is_train:
                dev_out = fmodel(dev_features)
                outer_train_loss = dev_out["loss"]
                output_dict = {
                    'inner_loss': inner_train_loss,
                    'outer_loss': outer_train_loss
                }

            else:
                with torch.no_grad:
                    dev_out = self.model(dev_features)
                    output_dict = {
                        'inner_loss': inner_train_loss,
                        'outer_loss': outer_train_loss,
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
        for mkey in output_dict:
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
            self.hparams.train_dir,
            data_type="train", is_training=True
        )
        self.dev_data = MetaKnowledgeDataset(
            self.model_logger,
            self.hparams,
            self.hparams.train_dir,
            data_type="dev",
            is_training=False
        )

        self.train_data.load_dataset(self.tokenizer)
        self.dev_data.load_dataset(self.tokenizer)
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


def generate(
    model,
    input_ids,
    attention_mask,
    max_length=4,
    no_repeat_ngram_size=None,
    num_beams=10,
    do_sample=None,
    top_p=None,
    min_length=1,
    top_k=None,
    num_return_sequences=None
):
    if do_sample and top_p:
        top_k = 0
    elif do_sample and top_k:
        top_p = None

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        use_cache=True
    )
    return outs


def inference(model, tokenizer, eval_dataloader):
    model.eval()
    all_out = []
    all_label = []
    for batch in tqdm(eval_dataloader, desc="Inference"):
        batch = batch[0]
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(model.parameters(), lr=1e-1)

        with higher.innerloop_ctx(model, inner_opt, track_higher_grads=False) as (fmodel, diffopt):
            for _ in range(n_inner_iter):
                train_out = fmodel(
                    input_ids=batch["train_input_ids"].to(
                        torch.device("cuda")),
                    attention_mask=batch["train_attention_mask"].to(
                        torch.device("cuda")),
                    labels=batch["train_labels"].to(torch.device("cuda"))
                )
                train_loss = train_out.loss
                diffopt.step(train_loss)

            with torch.no_grad():
                output = generate(
                    fmodel,
                    batch["input_ids"].to(torch.device("cuda")),
                    batch["attention_mask"].to(torch.device("cuda"))
                )
                eval_out = [tokenizer.decode(
                    out, skip_special_tokens=True) for out in output]
                eval_label = [tokenizer.decode(
                    label, skip_special_tokens=True) for label in batch["labels"]]
                all_out.extend(eval_out)
                all_label.extend(eval_label)

    return accuracy_score(all_label, all_out)


# @dataclass
# class TranslationOutput:
#     """Helper class for translation output"""
#     config: Dict
#     print_data: Dict

#     @classmethod
#     def from_output(cls, config, output):
#         """Loads from outputs

#         :param outputs: the outputs produced by the model
#         """
#         # retrieve data needed for printing
#         print_data = {}
#         print_out_keys = set(
#             itertools.chain(*[list(i["print_out"].keys()) for i in output])
#         )

#         for key_name in print_out_keys:
#             raw_data = [t["print_out"][key_name] for t in output]
#             print_data[key_name] = [t for t in itertools.chain(*raw_data)]

#         return cls(config=config, print_data=print_data)

#     @property
#     def prefixes(self):
#         return self.print_data.get("prefix", [])

#     @property
#     def label_scores(self):
#         return self.print_data.get("label_scores", [])

#     @property
#     def targets(self):
#         return self.print_data.get("text_out", [])

#     @property
#     def outputs(self):
#         return self.print_data.get("gen_out", [])

#     def _normalize_text(self, s):
#         """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
#         import string
#         import re

#         def remove_articles(text):
#             regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
#             return re.sub(regex, " ", text)

#         def white_space_fix(text):
#             return " ".join(text.split())

#         def remove_punc(text):
#             exclude = set(string.punctuation)
#             return "".join(ch for ch in text if ch not in exclude)

#         def lower(text):
#             return text.lower()

#         return white_space_fix(remove_articles(remove_punc(lower(s))))

#     def _compute_f1(self, generation_pairs):
#         prediction, truth = generation_pairs
#         pred_tokens = self._normalize_text(prediction).split()
#         truth_tokens = self._normalize_text(truth).split()

#         # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
#         if len(pred_tokens) == 0 or len(truth_tokens) == 0:
#             return int(pred_tokens == truth_tokens)

#         common_tokens = set(pred_tokens) & set(truth_tokens)

#         # if there are no common tokens then f1 = 0
#         if len(common_tokens) == 0:
#             return 0

#         prec = len(common_tokens) / len(pred_tokens)
#         rec = len(common_tokens) / len(truth_tokens)

#         return 2 * (prec * rec) / (prec + rec)

#     def _jaccard(self, str_pair):
#         str1, str2 = str_pair
#         a = set(str1.lower().split())
#         b = set(str2.lower().split())
#         if (len(a) == 0) & (len(b) == 0):
#             return 0.5
#         c = a.intersection(b)

#         return float(len(c)) / (len(a) + len(b) - len(c))

#     def gen_f1(self, sort=False):
#         """Returns an f1 score for generation

#         :rtype: float or None
#         """
#         targets = self.targets
#         outputs = self.outputs

#         targets = [self._normalize_text(t.strip())
#                    for k, t in enumerate(targets)]
#         outputs = [self._normalize_text(t.strip())
#                    for k, t in enumerate(outputs)]

#         if sort is True:
#             util_logger.info('Sorting output...')
#             targets = ["+".join(sorted(t.split("+"))) for t in targets]
#             outputs = ["+".join(sorted(o.split("+"))) for o in outputs]

#         if targets and outputs and len(targets) == len(outputs):
#             util_logger.info(
#                 'First few inputs: %s' % ', '.join(targets[:4])
#             )
#             util_logger.info(
#                 'First few outputs: %s' % ', '.join(outputs[:4])
#             )
#             batch_f1 = list(map(self._compute_f1, zip(targets, outputs)))
#             avg_f1 = sum(batch_f1) / len(batch_f1)
#             return avg_f1

#     def gen_jaccard(self, sort=False):
#         """Returns an jaccard text accuracy for generation

#         :rtype: float or None
#         """
#         targets = self.targets
#         outputs = self.outputs

#         targets = [self._normalize_text(t.strip())
#                    for k, t in enumerate(targets)]
#         outputs = [self._normalize_text(t.strip())
#                    for k, t in enumerate(outputs)]

#         if sort is True:
#             util_logger.info('Sorting output...')
#             targets = ["+".join(sorted(t.split("+"))) for t in targets]
#             outputs = ["+".join(sorted(o.split("+"))) for o in outputs]

#         if targets and outputs and len(targets) == len(outputs):
#             util_logger.info(
#                 'First few inputs: %s' % ', '.join(targets[:4])
#             )
#             util_logger.info(
#                 'First few outputs: %s' % ', '.join(outputs[:4])
#             )
#             batch_jaccard = list(map(self._jaccard, zip(targets, outputs)))
#             avg_jaccard = sum(batch_jaccard) / len(batch_jaccard)
#             return avg_jaccard

#     def gen_em(self, sort=False):
#         """Returns an exact match accuracy for generation

#         :rtype: float or None
#         """
#         prefixes = self.prefixes
#         targets = self.targets
#         outputs = self.outputs

#         if prefixes:
#             targets = [t.strip() for k, t in enumerate(targets)
#                        if not prefixes[k] or prefixes[k] == "answer:"]
#             outputs = [t.strip() for k, t in enumerate(outputs)
#                        if not prefixes[k] or prefixes[k] == "answer:"]

#         # special sorting functionality for set processing
#         # assumes set items are delimited by `+`
#         if sort is True:
#             util_logger.info('Sorting output...')
#             targets = ["+".join(sorted(t.split("+"))) for t in targets]
#             outputs = ["+".join(sorted(o.split("+"))) for o in outputs]

#         if targets and outputs and len(targets) == len(outputs):
#             util_logger.info(
#                 'First few inputs: %s' % ', '.join(targets[:4])
#             )
#             util_logger.info(
#                 'First few outputs: %s' % ', '.join(outputs[:4])
#             )
#             return sklearn_metrics.accuracy_score(targets, outputs)

#     @property
#     def generative(self):
#         return True

#     def enumerate_instances(self):
#         """Enumerate through instances for printing

#         """
#         guids = self.print_data["guid"]
#         text_in = self.print_data["text_in"]
#         prefixes = self.prefixes
#         targets = self.targets
#         outputs = self.outputs
#         label_scores = self.label_scores

#         total_outputs = []

#         for k, identifier in enumerate(guids):
#             instance_dict = {}
#             instance_dict["id"] = identifier
#             instance_dict["context"] = text_in[k]
#             instance_dict["gen_out"] = outputs[k]
#             if targets:
#                 instance_dict["answer"] = targets[k]
#             if prefixes:
#                 instance_dict["meta"] = {}
#                 instance_dict["meta"]["prefix"] = prefixes[k]
#             if label_scores:
#                 instance_dict["label_scores"] = label_scores[k]

#             total_outputs.append(instance_dict)

#         return total_outputs

#     def __iter__(self):
#         for item in self.enumerate_instances():
#             yield item
