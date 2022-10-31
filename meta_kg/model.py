import wandb
import itertools
import torch.nn as nn
import numpy as np

from collections import Counter
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from evaluate import load
from sklearn.metrics import accuracy_score
from transformers import TextGenerationPipeline

from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM
)

from meta_kg.utils.py_io import write_json

model_class_registry = {
    "t5": T5ForConditionalGeneration,
    "gpt2": GPT2LMHeadModel,
    "gptj": GPTJForCausalLM,
    "gpt-neo": GPTNeoForCausalLM,
}

exact_match_metric = load("exact_match")


class GeneratorModel():
    """Convenient class for implementing a standard generator model"""

    def generate(
        self,
        input_ids,
        attention_mask,
        max_length=None,
        min_length=None,
        num_beams=None,
        do_sample=None,
        top_p=None,
        top_k=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None
    ):
        """Calls the underlying model generator. Low-level function
        used during training and validation.

        :param features: the input features to the model
        :param max_length: the maximum length of the generated sequence
        :param min_length: the minimum length of the generated sequence
        :param num_beams: the number of beams to use for beam search
        :param do_sample: whether to use sampling or not
        :param top_p: the top-p value to use for sampling
        :param top_k: the top-k value to use for sampling
        :param no_repeat_ngram_size: the n-gram size to use for sampling
        :param num_return_sequences: the number of sequences to return

        :returns: the generated sequences
        """
        if do_sample and top_p:
            top_k = 0
        elif do_sample and top_k:
            top_p = None

        # note : doesn't require outputs
        # this is good to avoid issues with relative attention
        # https://github.com/huggingface/transformers/issues/10484

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            use_cache=True
        )
        return outputs


class PretrainedEncoderDecoder(nn.Module, GeneratorModel):
    """Generic transformer-based pretrained encoder decoder (e.g., T5, BART, etc..)
    which has the added feature of doing on-the-fly generation during training and evaluation.
    """

    def __init__(self, model, tokenizer, model_config, global_config):
        super().__init__()
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.global_config = global_config
        self.pipe = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=global_config.device_idx,
            return_full_text=False,
            max_new_tokens=32,
        )

    @classmethod
    def from_config(cls, config):
        """Loads a pretrained encoder decoder from configuration

        :param config: the global configuration
        """

        model_class = model_class_registry[config.model_type]

        model = model_class.from_pretrained(config.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path)
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)

        if "gpt" in config.model_type:
            tokenizer.add_special_tokens({
                'bos_token': '<|startoftext|>',
                'eos_token': '<|endoftext|>',
                'pad_token': '<pad>',
                'additional_special_tokens': ['<gen>']
            })

            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id

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

    def forward(self, features, print_out):
        """A modified version of forward method for the underlying
           `ConditionalSequenceGeneration` model. It combines loss
           measurement and generation into one step.
        :param features: the target inputs
        :param print_out: data to print out during evaluation
        """
        main_out = {"print_out": {}}
        main_out["print_out"].update(print_out)

        outputs = self.model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            labels=features["input_ids"],
            return_dict=True
        )

        main_out["loss"] = outputs.loss

        if "evaluate" in features and features["evaluate"]:
            if "question" in print_out:
                main_out["print_out"]["gen_out"] = [
                    self.pipe(q)[0]["generated_text"].strip() for q in main_out["print_out"]["question"]]
            else:
                main_out["print_out"]["gen_out"] = [
                    self.pipe(q)[0]["generated_text"].strip() for q in main_out["print_out"]["prompt"]]

        return main_out

    def output_parser_metrics(self, raw_output):
        """Function responsible for parsing the raw_output and computing particular
        metrics from the model runner output.

        :param raw_output: the raw output created by the model runner
        :rtype: tuple
        """
        metrics = {}
        sout = TranslationOutput.from_output(self.global_config, raw_output)

        scores = sout.gen_em()
        if isinstance(scores, dict):
            metrics.update(scores)
        else:
            metrics["acc"] = scores
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

    @ classmethod
    def from_output(cls, config, output):
        """Loads from raw outputs

        :param outputs: the outputs produced by the model
        """

        print_data = cls.get_print_data(output, "print_out")

        return cls(config=config, print_data=print_data)

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

    def gen_em(self):
        """Returns an exact match accuracy for generation

        :rtype: float or None
        """
        targets = self.targets
        outputs = self.outputs

        class_count = Counter(targets)

        acc_class = {
            "true": 0,
            "false": 0,
            "unknown": 0,
        }

        if targets and outputs and len(targets) == len(outputs):
            for label, gen in zip(targets, outputs):
                if gen.strip() == label:
                    acc_class[label] += 1
                
            for key in acc_class:
                if class_count[key] > 0:
                    acc_class[key] /= class_count[key]
                else:
                    acc_class[key] = 0

            acc = accuracy_score(targets, outputs)

            return {
                "acc": acc,
                "acc_true": acc_class["true"],
                "acc_false": acc_class["false"],
                "acc_unknown": acc_class["unknown"],
            }

    @ property
    def generative(self):
        return True

    def enumerate_instances(self):
        """Enumerate through instances for printing

        """
        guids = self.print_data["guid"]
        text_in = self.print_data["question"]
        targets = self.targets
        outputs = self.outputs
        inner_loss = self.print_data["inner_loss"]

        if "inner_print_out" in self.print_data:
            inner_out = self.print_data["inner_print_out"]
        else:
            inner_out = None

        total_outputs = []

        for k, identifier in enumerate(guids):
            instance_dict = {}
            instance_dict["guid"] = identifier
            instance_dict["question"] = text_in[k]
            instance_dict["gen_out"] = outputs[k]
            instance_dict["answer"] = targets[k]
            instance_dict["inner_loss"] = inner_loss[k]
            if inner_out:
                instance_dict["inner_out"] = inner_out[k]
            
            # if prefixes:
            #     instance_dict["meta"] = {}
            #     instance_dict["meta"]["prefix"] = prefixes[k]
            # if label_scores:
            #     instance_dict["label_scores"] = label_scores[k]

            total_outputs.append(instance_dict)

        return total_outputs

    def __iter__(self):
        for item in self.enumerate_instances():
            yield item
