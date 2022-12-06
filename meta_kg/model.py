import re
import string
import torch
import wandb
import itertools
import numpy as np
import torch.nn as nn

from collections import Counter
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
)

from meta_kg.utils.py_io import write_json
from meta_kg.dataset import dataset_config

model_class_registry = {
    "t5": AutoModelForSeq2SeqLM,
    "gpt2": GPT2LMHeadModel,
    "gptj": AutoModelForCausalLM,
    "gpt-neo": AutoModelForCausalLM,
}


class PretrainedEncoderDecoder(nn.Module):
    """Generic transformer-based pretrained encoder decoder (e.g., T5, BART, etc..)
    which has the added feature of doing on-the-fly generation during training and evaluation.
    """

    def __init__(self, model, tokenizer, model_config, global_config):
        super().__init__()
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.global_config = global_config

        self.labels = dataset_config[global_config.dataset_type]["labels"]
        self.id_to_label = dataset_config[global_config.dataset_type]["id_to_label"]

        # self.lm_head_inner = nn.Linear(
        #     model_config.n_embd,
        #     model_config.vocab_size,
        #     bias=False
        # )
        # self.lm_head_inner.load_state_dict(
        #     self.model.lm_head.state_dict()
        # )
        # self.lm_head_inner.requires_grad = False

    @classmethod
    def from_config(cls, config):
        """Loads a pretrained encoder decoder from configuration

        :param config: the global configuration
        """
        model_class = model_class_registry[config.model_type]
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        model = model_class.from_pretrained(config.model_name_or_path)
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        model.config.pad_token_id = tokenizer.eos_token_id

        if config.freeze_partial:
            for name, param in model.named_parameters():
                if np.any([x in name for x in [
                    '.0.', '.1.', '.2.', '.3.',
                    '.4.', '.5.', '.6.', '.7.',
                ]]):
                    print(f"Freezing {name}")
                    param.requires_grad = False

        return cls(
            model,
            tokenizer,
            model_config,
            config,
        )

    def forward(self, features, print_out, is_inner=False):
        """A modified version of forward method for the underlying
           `ConditionalSequenceGeneration` model. It combines loss
           measurement and generation into one step.
        :param features: the target inputs
        :param print_out: data to print out during evaluation
        """
        main_out = {"print_out": print_out}
        labels = features["input_ids"] if "gpt" in self.global_config.model_type else features["labels"]

        outputs = self.model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        # if is_inner:
        #     hidden_states = outputs["hidden_states"][-1]
        #     lm_logits = self.lm_head_inner(hidden_states)
        # else:
        lm_logits = outputs["logits"]

        logits = lm_logits[..., :-1, :].contiguous()
        labels = features["input_ids"][..., 1:].contiguous()
        label_mask = features["token_type_ids"][..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        loss = torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
        main_out["loss"] = loss.mean()

        if is_inner and self.global_config.inner_verbose:
            main_out["token_loss"] = []
            for i in range(losses.size(0)):
                main_out["token_loss"].append(
                    self.compute_token_loss(losses[i], labels[i]*label_mask[i]))
            norm_logits = torch.softmax(logits, dim=-1)
            batch_topk_tokens = self.get_tok_tokens(
                norm_logits, labels, label_mask)
            main_out["print_out"]["topk_tokens"] = [batch_topk_tokens]

        if "evaluate" in features and features["evaluate"]:
            if "question" in print_out:
                main_out["print_out"]["gen_out"] = self.generate(print_out)
            else:
                main_out["print_out"]["gen_out"] = self.generate(
                    main_out["print_out"]["prompt"])
        return main_out

    def generate(self, print_out):
        device = self.model.device

        output_length = []
        for answer in print_out["answer"]:
            out_ids = self.tokenizer(answer, return_tensors="pt").input_ids
            output_length.append(out_ids.size(1))
        max_out_length = max(output_length)

        input_ids_batch = []
        for question in print_out["question"]:
            input_ids = self.tokenizer(
                question, return_tensors="pt").input_ids.to(device)
            input_ids_batch.append(input_ids)

        outputs = []
        for input_ids in input_ids_batch:
            max_length = input_ids.size(1) + max_out_length
            greedy_output = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                top_p=None,
                top_k=5,
                do_sample=False,
                num_return_sequences=1,
                use_cache=True)
            out = self.tokenizer.decode(
                greedy_output[0],
                skip_special_tokens=True)
            outputs.append(out)

        return outputs

    def get_tok_tokens(self, norm_logits, input_ids, label_mask):
        batch_topk_tokens = []
        for i in range(norm_logits.size(0)):
            input_ids_item = input_ids[i]
            norm_logits_item = norm_logits[i] * label_mask[i].unsqueeze(-1)
            prob_locs = torch.nonzero(norm_logits_item)[:, 0].unique()
            topk_tokens = self.get_token_prob(
                prob_locs, norm_logits_item, input_ids_item)
            batch_topk_tokens.append(topk_tokens)
        return batch_topk_tokens

    def get_token_prob(self, prob_locs, norm_logits, input_ids, k=10):
        topk_tokens_seq = []
        for loc in prob_locs:
            token = self.tokenizer.decode(input_ids[loc])
            topk_prob = torch.topk(norm_logits[loc], k=k)
            topk_tokens = []
            for idx in topk_prob.indices:
                gen_token = self.tokenizer.decode(idx.unsqueeze(0))
                prob = round(norm_logits[loc][idx].item() * 100, 3)
                topk_tokens.append((token.strip(), gen_token.strip(), prob))
            topk_tokens_seq.append(topk_tokens)
        return topk_tokens_seq

    def compute_token_loss(self, loss, input_ids):
        """Computes the token-level loss for the given loss
        :param loss: the sentence-level LM loss
        :param input_ids: the input ids
        :return: the token-level LM loss
        """
        matched_loss = list(zip(loss.cpu().detach(), input_ids))
        token_loss = {}
        for i, (loss, token_id) in enumerate(matched_loss):
            if token_id > 0:
                token = self.tokenizer.decode(token_id).strip()
                token_loss[f"{i}: {token}"] = loss.item()

        return token_loss

    def output_parser_metrics(self, raw_output):
        """Function responsible for parsing the raw_output and computing particular
        metrics from the model runner output.

        :param raw_output: the raw output created by the model runner
        :rtype: tuple
        """
        metrics = {}
        sout = TranslationOutput.from_output(
            self.global_config, raw_output, self.labels)
        scores = sout.compute_metrics()
        class_scores = sout.fine_grained_metrics()
        metrics.update(scores)
        metrics.update(class_scores)
        return (sout, metrics)

    def evaluate_output(self, output, out_file=None, metric_file=None, is_test=False):
        """Method for generating output produced during training and/or evaluation.

        :param output: the output generated by runner
        :raises: ValueError
        """
        sout, metrics = self.output_parser_metrics(output)
        if out_file:
            out_dir = Path(out_file).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs = []
            for instance in sout:
                outputs.append(instance)
            write_json(outputs, out_file)

            if is_test:
                artifact = wandb.Artifact(f"test_eval_out", type='dataset')
                artifact.add_file(out_file)
                wandb.run.log_artifact(artifact)

        if metric_file:
            out_dir = Path(metric_file).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(metrics, metric_file)

            if is_test:
                artifact = wandb.Artifact(f"test_metrics", type='dataset')
                artifact.add_file(out_file)
                wandb.run.log_artifact(artifact)

        return metrics


@ dataclass
class TranslationOutput:
    """Helper class for translation output"""
    config: Dict
    print_data: Dict
    labels: List[str]

    @ classmethod
    def from_output(cls, config, output, labels=None):
        """Loads from raw outputs

        :param outputs: the outputs produced by the model
        """

        print_data = cls.get_print_data(output, "print_out")

        return cls(config=config, print_data=print_data, labels=labels)

    @ classmethod
    def get_print_data(cls, output, print_key):
        print_data = {}
        print_out_keys = set(
            itertools.chain(*[list(i[print_key].keys()) for i in output])
        )

        for key_name in print_out_keys:
            raw_data = [t[print_key][key_name]
                        if key_name in t[print_key] else [] for t in output]
            print_data[key_name] = [t for t in itertools.chain(*raw_data)]

        inner_outs = []
        for item in output:
            if "inner_print_out" in item:
                inner_outs.append(item["inner_print_out"])

        if len(inner_outs) > 0:
            print_data["inner_print_out"] = inner_outs

        return print_data

    @ property
    def prefixes(self):
        return self.print_data.get("prefix", [])

    @ property
    def questions(self):
        return self.print_data.get("question", [])

    @ property
    def targets(self):
        return self.print_data.get("answer", [])

    @ property
    def outputs(self):
        return self.print_data.get("gen_out", [])

    def fine_grained_metrics(self):
        """Computes the fine grained metrics for the output"""
        targets = self.targets
        outputs = self.outputs
        class_count = Counter(targets)

        class_metrics = {}
        for c in self.labels:
            if c not in class_count:
                continue
            class_metrics[c] = {}
            class_metrics[c]['acc'] = 0
            class_metrics[c]['f1'] = 0

        metrics = {}
        if targets and outputs and len(targets) == len(outputs):
            for label, gen in zip(targets, outputs):
                em = self.compute_exact_match(gen, label)
                f1 = self.compute_f1(gen, label)
                if label in class_metrics:
                    class_metrics[label]['acc'] += em
                    class_metrics[label]['f1'] += f1

            for label in class_metrics:
                for metric in class_metrics[label]:
                    total_score = class_metrics[label][metric]
                    metrics[f"{metric}_{label}"] = total_score / \
                        class_count[label]

        return metrics

    def compute_metrics(self):
        """Returns an exact match accuracy for generation

        :rtype: float or None
        """
        targets = self.targets
        outputs = self.outputs

        metrics = {}
        if targets and outputs and len(targets) == len(outputs):
            em_scores = [self.compute_exact_match(
                gen, label) for label, gen in zip(targets, outputs)]
            f1_scores = [self.compute_f1(gen, label)
                         for label, gen in zip(targets, outputs)]

            metrics["acc"] = sum(em_scores) / len(targets)
            metrics["f1"] = sum(f1_scores) / len(targets)

        return metrics

    def normalize_text(self, text):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))

    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(truth) in self.normalize_text(prediction))

    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)

    @ property
    def generative(self):
        return True

    def enumerate_instances(self):
        """Enumerate through instances for printing

        """
        guids = self.print_data["guid"]
        prefixes = self.prefixes

        text_in = self.print_data["question"]
        targets = self.targets
        outputs = self.outputs

        inner_loss_prev = self.print_data.get("inner_loss_prev")
        inner_loss = self.print_data.get("inner_loss")
        inner_loss_token_prev = self.print_data.get("inner_loss_token_prev")
        inner_loss_token = self.print_data.get("inner_loss_token")
        inner_loss_diff = self.print_data.get("inner_loss_diff")
        inner_out = self.print_data.get("inner_print_out")
        topk_tokens = self.print_data.get("topk_tokens")

        total_outputs = []
        for k, identifier in enumerate(guids):
            instance_dict = {}
            instance_dict["guid"] = identifier
            instance_dict["prefix"] = prefixes[k]
            instance_dict["question"] = text_in[k]
            instance_dict["gen_out"] = outputs[k]
            instance_dict["answer"] = targets[k]

            if inner_loss_prev:
                instance_dict["inner_loss_prev"] = inner_loss_prev[k]
            if inner_loss:
                instance_dict["inner_loss"] = inner_loss[k]
            if inner_loss_token_prev:
                instance_dict["inner_loss_token_prev"] = inner_loss_token_prev[k]
            if inner_loss_diff:
                instance_dict["inner_loss_diff"] = inner_loss_diff[k]
            if inner_loss_token:
                instance_dict["inner_loss_token"] = inner_loss_token[k]
            if inner_out:
                instance_dict["inner_out"] = inner_out[k]
            if topk_tokens:
                instance_dict["topk_tokens"] = topk_tokens[k]

            total_outputs.append(instance_dict)

        return total_outputs

    def __iter__(self):
        for item in self.enumerate_instances():
            yield item
