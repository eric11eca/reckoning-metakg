import os
import re
import json
import uuid
import string
import torch
import numpy as np

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from .reader import JsonlReader
from .utils import read_jsonl


@dataclass
class MetaQAInput:
    """Base class for input examples

    :param text: the input text
    :param guid: the instance identifier (optional)
    """
    guid: str = field(default='')
    qa_pairs: List[Tuple] = field(default_factory=list)
    facts: List[Tuple] = field(default_factory=list)
    evaluation: bool = False
    prefix: str = ""

    def __str__(self):
        return "<InputExample> label: {}, qa_pairs: {}, facts: {}, evaluation={}, prefix={}".format(
            self.label, self.question,
            self.answer, str(self.evaluation), self.prefix,
        )

    @classmethod
    def from_json(cls, json_instance):
        qa_pairs = json_instance["qa_pairs"] if "qa_pairs" in json_instance else None
        facts = json_instance["facts"] if "facts" in json_instance else None
        guid = json_instance["guid"] if "guid" in json_instance else None
        return cls(guid, qa_pairs, facts)


class MetaQADataReader():
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"] if "guid" in instance else str(uuid.uuid1())
        qa_pairs = instance["qa_pairs"]
        facts = instance["facts"]

        return MetaQAInput(
            guid=guid,
            qa_pairs=qa_pairs,
            facts=facts
        )

    @classmethod
    def jsonl_file_reader(cls, path, evaluation):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param evaluation: indicator as to where this is an evaluation file (based on name)
        :param config: the configuration
        """
        total_data = read_jsonl(path)

        total_qa_data = []
        for instance in total_data:
            qa_data = cls._read(instance)
            qa_data.evaluation = evaluation
            total_qa_data.append(qa_data)

        return total_qa_data


def meta_qa_data_loader(
    data_path: str,
    split: str,
    evaluate=False
):
    """Code for loading data and creating dataset cache

    :param data_path: the location of the data
    :param split: the data split
    :param config: the global configuration
    :param evaluate: whether or not this is an evaluation split
    :param tokenizer: the model tokenizer, for possibly checking truncation
    """
    target_data = os.path.join(data_path, split+".jsonl")
    # logger.info(
    #    f'Reading data from {target_data}, evaluate={str(evaluate)}')

    data_container = MetaQADataReader.jsonl_file_reader(
        target_data,
        evaluate=evaluate
    )

    return data_container


class MetaKnowledgeDataset(object):
    def __init__(self, logger, args, data_path, data_type, is_training):
        self.data_path = data_path
        self.data_type = data_type
        self.task_name = args.dataset
        self.data = read_jsonl(
            f"{data_path}/{args.dataset}/{data_type}.jsonl")

        # self.meta_qa_data = meta_qa_data_loader(
        #     data_path,
        #     split="train",
        #     config=args,
        #     evaluate=False
        # )

        # for qa_data in self.meta_qa_data:
        #     train_examples = qa_data["facts"]
        #     dev_examples = qa_data["qa_pairs"]

        #     self.data.append({
        #         "guid": qa_data["guid"],
        #         "train_examples": train_examples,
        #         "dev_examples": dev_examples
        #     })

        self.is_training = is_training
        self.logger = logger
        self.args = args
        self.metric = "acc"
        self.tokenizer = None
        self.dataset = []
        self.dataloader = None
        self.cache = None
        self.load = not args.debug
        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer

        preprocessed_path = os.path.join(
            *[self.data_path, f"{self.args.dataset}", f"{self.data_type}.pt"]
        )

        if self.load and os.path.exists(preprocessed_path):
            self.logger.info(
                "Loading pre-tokenized data from {}".format(preprocessed_path))
            self.dataset = torch.load(preprocessed_path)

        else:
            self.logger.info(
                "Start tokenizing ... {} instances".format(len(self.data)))

            for qa_data in self.data:
                train_facts = []
                for fact in qa_data["facts"]:
                    train_tokenized_input = tokenizer(
                        fact[0],
                        truncation=True,
                        return_tensors="pt",
                        max_length=128
                    )

                    train_tokenized_output = tokenizer(
                        fact[1],
                        truncation=True,
                        return_tensors="pt",
                        max_length=16
                    )

                    fact_batch = {
                        "input_ids": train_tokenized_input["input_ids"],
                        "attention_mask": train_tokenized_input["attention_mask"],
                        "labels": train_tokenized_output["input_ids"],
                    }

                    train_facts.append(fact_batch)

                if self.args.expand_dev:
                    for qa_pair in qa_data["qa_pairs"]:
                        dev_inputs = qa_pair[0]
                        dev_outputs = str(qa_pair[1])

                        dev_tokenized_input = tokenizer(
                            dev_inputs,
                            truncation=True,
                            return_tensors="pt",
                            max_length=128
                        )

                        dev_tokenized_output = tokenizer(
                            dev_outputs,
                            truncation=True,
                            return_tensors="pt",
                            max_length=16
                        )

                        feature = {
                            "input_ids": dev_tokenized_input["input_ids"],
                            "attention_mask": dev_tokenized_input["attention_mask"],
                            "labels": dev_tokenized_output["input_ids"],
                            "facts": train_facts
                        }

                        self.dataset.append(feature)

                else:
                    dev_inputs = [qa_pair[0]
                                  for qa_pair in qa_data["qa_pairs"]]
                    dev_outputs = [str(qa_pair[1])
                                   for qa_pair in qa_data["qa_pairs"]]

                    dev_tokenized_input = tokenizer.batch_encode_plus(
                        dev_inputs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=128
                    )

                    dev_tokenized_output = tokenizer.batch_encode_plus(
                        dev_outputs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=16
                    )

                    feature = {
                        "input_ids": dev_tokenized_input["input_ids"],
                        "attention_mask": dev_tokenized_input["attention_mask"],
                        "labels": dev_tokenized_output["input_ids"],
                        "facts": train_facts
                    }

                    self.dataset.append(feature)

            torch.save(self.dataset, preprocessed_path)

            self.logger.info("Loaded {} examples from {} data".format(
                len(self.dataset), self.data_type))

            if do_return:
                return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MetaDataLoader(
            self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def save_predictions(self, predictions):
        assert len(predictions) == len(self), (len(predictions), len(self))

        predictions = ['n/a' if len(prediction.strip()) ==
                       0 else prediction for prediction in predictions]
        prediction_text = [
            prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(self.args.output_dir,
                                 "{}_predictions.txt".format(self.args.prefix))
        with open(save_path, "w") as f:
            f.writelines(prediction_text)

        self.logger.info("Saved prediction in {}".format(save_path))


class MetaDataLoader(DataLoader):
    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        super(MetaDataLoader, self).__init__(
            dataset, sampler=sampler, batch_size=batch_size)
        self.collate_fn = self.dummy_collate
        self.args = args

    def dummy_collate(self, input_data):
        return input_data

    def inference_dataloader(self):
        bsz = self.args.predict_batch_size
        for idx, (start_idx, end_idx) in enumerate(self.dataset.metadata_rel):
            input_ids_for_this_rel = self.dataset.input_ids[start_idx: end_idx]
            masks_for_this_rel = self.dataset.attention_mask[start_idx: end_idx]
            for j in range(0, len(input_ids_for_this_rel), bsz):
                input_ids_this_batch = input_ids_for_this_rel[j: j+bsz]
                masks_for_this_batch = masks_for_this_rel[j: j+bsz]

                yield self.dataset.relation_ids[idx], self.dataset.relation_mask[idx], input_ids_this_batch, masks_for_this_batch


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([f1_score(prediction, gt) for gt in groundtruth])
    return f1_score(prediction, groundtruth)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
