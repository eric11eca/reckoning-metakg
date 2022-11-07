import os
import re
import random
import uuid
import string
import torch
import numpy as np

from pprint import pprint
from collections import Counter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .utils.py_io import read_jsonl
from .utils.datastructure import labels_to_bimap

dataset_config = {
    "clutrr1": {
        "labels": ["yes", "no", "maybe"],
        "label_to_id": {"yes": 0, "no": 1, "maybe": 2},
        "id_to_label": {0: "yes", 1: "no", 2: "maybe"},
        "num_labels": 3
    },
    "proofwriter_owa_natlang": {
        "labels": ["true", "false", "unknown"],
        "label_to_id": {"true": 0, "false": 1, "unknown": 2},
        "id_to_label": {0: "true", 1: "false", 2: "unknown"},
        "num_labels": 3
    },
    "proofwriter_cwa_natlang": {
        "labels": ["true", "false"],
        "label_to_id": {"true": 0, "false": 1},
        "id_to_label": {0: "true", 1: "false"},
        "num_labels": 2
    },
}


def kg_as_span_reconstruction(triples, rules):
    facts = []
    for kg in triples.values():
        text = kg['text']
        mask = kg['representation'].split('" "')[-2].strip()
        facts.append((text.replace(mask, "<extra_id_0>"),
                      f"<extra_id_0> {mask} <extra_id_1>"))
    for kg in rules.values():
        text = kg['text']
        mask = kg['representation'].split('" "')[-2].strip()
        facts.append((text.replace(mask, "<extra_id_0>"),
                      f"<extra_id_0> {mask} <extra_id_1>"))
    return facts


def kg_as_autoregressive(triples, rules, prefix="new fact: "):
    facts = [(prefix, kg['text']) for kg in triples.values()]
    facts.extend([(prefix, kg['text']) for kg in rules.values()])
    return facts


class ProofWriterDataReader():
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """

        questions = instance["questions"]
        triples = instance["triples"]
        rules = instance["rules"]
        guid = str(uuid.uuid4())

        if dataset_config[args.dataset] == 3:
            prefix = "Is it true, false, or unknown that"
        elif dataset_config[args.dataset] == 2:
            prefix = "Is it true or false that"

        qa_pairs = []
        unknown_paris = []
        for qa_item in questions.values():
            question = qa_item["question"].replace(".", "")
            question = f"{prefix} {question}?"
            answer = str(qa_item["answer"]).lower()
            if answer != "unknown":
                qa_pairs.append((question, answer))
            else:
                unknown_paris.append((question, answer))

        if len(unknown_paris) > 0:
            qa_pairs += random.choices(unknown_paris,
                                       k=len(unknown_paris) // 2)

        # qa_pairs += unknown_paris

        if args.input_format == "mlm":
            facts = kg_as_span_reconstruction(triples, rules)
        else:
            facts = []
            for item in qa_pairs:
                question = item[0].replace(f"{prefix} ", "")
                question = question.replace("?", "")
                facts.append(kg_as_autoregressive(
                    triples, rules,
                    prefix=f"To determine if {question}, a person needs to know"
                ))

        # assert len(facts) == len(qa_pairs) * (len(triples) + len(rules))

        return [{"guid": guid, "qa_pairs": [qa_pairs[i]], "facts": facts[i]} for i in range(len(qa_pairs))]

    @classmethod
    def jsonl_file_reader(cls, path, config):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param evaluation: indicator as to where this is an evaluation file (based on name)
        :param config: the configuration
        """
        total_data = read_jsonl(path)

        total_qa_data = []
        for instance in total_data:
            qa_data = cls._read(instance, config)

            total_qa_data += qa_data

        return total_qa_data

class ClutrrDataReader():
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """

        questions = instance["questions"]
        story = instance["facts"]
        guid = str(uuid.uuid4())
        prefix = "Is it true that"
        
        qa_pairs = []
        # unknown_paris = []
        for qa_item in questions:
            question = qa_item[0].replace(".", "")
            question = f"{prefix} {question}?"
            answer = str(qa_item[1]).lower()
            qa_pairs.append((question, answer))
            # if answer != "unknown":
            #     qa_pairs.append((question, answer))
            # else:
            #     unknown_paris.append((question, answer))
    
        facts = []
        for item in qa_pairs:
            question = item[0].replace(f"{prefix} ", "")
            question = question.replace("?", "")
            facts.append([(
                f"To determine if {question}, a person needs to know",
                fact) for fact in story])

        return [{"guid": guid, "qa_pairs": [qa_pairs[i]], "facts": facts[i]} for i in range(len(qa_pairs))]

    @classmethod
    def jsonl_file_reader(cls, path, config):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param evaluation: indicator as to where this is an evaluation file (based on name)
        :param config: the configuration
        """
        total_data = read_jsonl(path)

        total_qa_data = []
        for instance in total_data:
            qa_data = cls._read(instance, config)

            total_qa_data += qa_data
        
        for data in total_qa_data[:3]:
            pprint(data)

        return total_qa_data


def meta_qa_data_loader(
    task_name: str,
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
    reader  = {
        "proofwriter_owa_natlang": ProofWriterDataReader,
        "proofwriter_cwa_natlang": ProofWriterDataReader,
        "clutrr1": ClutrrDataReader
    }

    target_data = os.path.join(data_path, split+".jsonl")
    data_container = reader[task_name].jsonl_file_reader(
        target_data,
        evaluate=evaluate
    )

    return data_container


class MetaKnowledgeDataset(object):
    def __init__(self, logger, args, tokenizer, data_path, data_type, is_training):
        self.data_path = data_path
        self.data_type = data_type
        self.task_name = args.dataset
        self.args = args
        
        reader_classes = {
            "proofwriter_owa_natlang": ProofWriterDataReader,
            "proofwriter_cwa_natlang": ProofWriterDataReader,
            "clutrr1": ClutrrDataReader
        }
        self.reader = reader_classes[self.task_name]
        self.is_training = is_training
        self.logger = logger
        self.tokenizer = tokenizer
        self.dataloader = None
        self.load = False

        self.data = self.read_data_from_file()

    def __len__(self):
        return len(self.data)

    def read_data_from_file(self):
        file_path = f"{self.data_path}/{self.task_name}/{self.data_type}.jsonl"
        file_data = self.reader.jsonl_file_reader(file_path, self.args)
        return file_data

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataloader(self):
        meta_dataloader = MetaDataLoader(
            self.args,
            self.data,
            self.tokenizer,
            self.is_training
        )

        self.dataloader = meta_dataloader.dataloader

        return self.dataloader


class MetaDataLoader():
    def __init__(self, args, dataset, tokenizer, is_training):
        self.args = args
        self.task = args.dataset
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.evaluate = not is_training
        self.num_classes = dataset_config[self.task]["num_labels"]
        self.label_to_id = dataset_config[self.task]["label_to_id"]

        if is_training:
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        if args.input_format == "t2t":
            collate_fn = self.text2text_collator
        else:
            collate_fn = self.causal_lm_collator

        if args.baseline:
            collate_fn = self.causal_lm_base_collator

        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

    def _tensorize(self, input_txt, output_txt, max_length):
        """Converts a list of strings into a tensor of ids

        :param input_txt: the input text
        :param output_txt: the output text
        :param max_length: the maximum length of the sequence
        :return: the tensorized input and output
        """
        pad = self.tokenizer.pad_token
        pad_id = self.tokenizer.encode(pad)

        ids1 = self.tokenizer(input_txt, return_tensors="pt")["input_ids"]
        ids2 = self.tokenizer(output_txt, return_tensors="pt")["input_ids"]

        n_mask = max_length - ids1.size(1) - ids2.size(1)
        assert n_mask >= 0, (max_length, ids1.size(1), ids2.size(1))
        padding = torch.LongTensor(pad_id * n_mask).unsqueeze(0)

        input_ids = torch.cat((ids1, ids2, padding), dim=1)
        attention_mask = torch.LongTensor(
            [1] * (ids1.size(1) + ids2.size(1)) + [0] * n_mask).unsqueeze(0)
        token_type_ids = torch.LongTensor(
            [0] * ids1.size(1) + [1] * ids2.size(1) + [0] * n_mask).unsqueeze(0)

        assert input_ids.size(1) == attention_mask.size(
            1) == token_type_ids.size(1) == max_length

        return input_ids, attention_mask, token_type_ids

    def causal_lm_base_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        eos = self.tokenizer.eos_token
        bos = self.tokenizer.bos_token

        facts_batch = [[fact[1] for fact in data['facts']] for data in batch]
        questions = [f"{data['qa_pairs'][0][0]}" for data in batch]
        answers = [data['qa_pairs'][0][1] for data in batch]

        max_length = 600

        input_ids_batch, attention_mask_batch, token_type_ids_batch = [], [], []
        for facts, question, answer in zip(facts_batch, questions, answers):
            fact_prefix = f"{bos}"
            for fact in facts:
                fact_prefix += f"{fact} "

            input_txt = f"{fact_prefix}\n{question}"
            output_txt = f"{answer}{eos}"

            input_ids, attention_mask, token_type_ids = self._tensorize(
                input_txt, output_txt, max_length)

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            token_type_ids_batch.append(token_type_ids)

        labels = torch.LongTensor([self.label_to_id[x] for x in answers])
        labels = torch.nn.functional.one_hot(
            labels, 
            num_classes=self.num_classes
        ).to(torch.float)
        

        print_inputs = [data['qa_pairs'][0][0] for data in batch]
        print_outputs = [data['qa_pairs'][0][1] for data in batch]
        print_out = {
            "guid": [data['guid'] for data in batch],
            "prefix": [self.args.dataset for data in batch],
            "question": print_inputs,
            "answer": print_outputs,
        }

        feature = {
            "input_ids": torch.cat(input_ids_batch, dim=0),
            "attention_mask": torch.cat(attention_mask_batch, dim=0),
            "token_type_ids": torch.cat(token_type_ids_batch, dim=0),
            "labels": labels,
            "print_out": print_out,
            "evaluate": self.evaluate
        }

        return feature

    def causal_lm_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        eos = self.tokenizer.eos_token
        bos = self.tokenizer.bos_token
        qa_data = batch[0]
        max_length = 48

        train_input_ids_batch = []
        train_attention_mask_batch = []
        train_token_type_ids_batch = []

        for i, fact in enumerate(qa_data["facts"]):
            # train_input_txt = f"{bos}{fact[0]} fact_{i}: {fact[1]}"
            train_input_txt = f"{bos}fact_{i} is {fact[1]}"
            train_output_txt = f"{fact[1]}{eos}"
            input_ids, attention_mask, token_type_ids = self._tensorize(
                train_input_txt, train_output_txt, max_length)
            train_input_ids_batch.append(input_ids)
            train_attention_mask_batch.append(attention_mask)
            train_token_type_ids_batch.append(token_type_ids)

        dev_input_ids_batch = []
        dev_attention_mask_batch = []
        dev_token_type_ids_batch = []
        labels_batch = []

        for qa_pair in qa_data["qa_pairs"]:
            dev_input_txt = f"{bos}{qa_pair[0]}"
            dev_output_txt = f"{qa_pair[1]}{eos}"
            input_ids, attention_mask, token_type_ids = self._tensorize(
                dev_input_txt, dev_output_txt, max_length)
            dev_input_ids_batch.append(input_ids)
            dev_attention_mask_batch.append(attention_mask)
            dev_token_type_ids_batch.append(token_type_ids)
            labels = torch.nn.functional.one_hot(
                torch.tensor([self.label_to_id[str(qa_pair[1])]]), 
                num_classes=self.num_classes
            ).to(torch.float)
            labels_batch.append(labels)

        feature = {
            "input_ids": torch.cat(dev_input_ids_batch, dim=0),
            "attention_mask": torch.cat(dev_attention_mask_batch, dim=0),
            "token_type_ids": torch.cat(dev_token_type_ids_batch, dim=0),
            "labels": torch.cat(labels_batch, dim=0),
            "train_input_ids": torch.cat(train_input_ids_batch, dim=0),
            "train_attention_mask": torch.cat(train_attention_mask_batch, dim=0),
            "train_token_type_ids": torch.cat(train_token_type_ids_batch, dim=0),
            "print_out": {"guid": [qa_data["guid"]]},
            "evaluate": self.evaluate
        }

        if self.evaluate:
            train_inputs = [
                fact[0] for fact in qa_data["facts"]]
            train_outputs = [
                fact[1] for fact in qa_data["facts"]]
            dev_inputs_eval = [
                f"{bos}{qa_pair[0]}" for qa_pair in qa_data["qa_pairs"]]
            dev_outputs = [str(qa_pair[1])
                           for qa_pair in qa_data["qa_pairs"]]

            feature["print_out"].update({
                "question": dev_inputs_eval,
                "answer": dev_outputs,
                "prefix": [self.args.dataset],
            })

            feature["inner_print_out"] = {
                "prompt": train_inputs,
                "fact": train_outputs,
                "guid": [qa_data["guid"]]
            }

        return feature

    def text2text_collator(
        self,
        batch,
    ):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        qa_data = batch[0]
        train_inputs = [fact[0] for fact in qa_data["facts"]]
        train_outputs = [fact[1] for fact in qa_data["facts"]]

        train_tokenized_input = self.tokenizer.batch_encode_plus(
            train_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=32
        )

        train_tokenized_output = self.tokenizer.batch_encode_plus(
            train_outputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=16
        )

        dev_inputs = [qa_pair[0]
                      for qa_pair in qa_data["qa_pairs"]]
        dev_outputs = [str(qa_pair[1])
                       for qa_pair in qa_data["qa_pairs"]]

        dev_tokenized_input = self.tokenizer.batch_encode_plus(
            dev_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=64
        )

        dev_tokenized_output = self.tokenizer.batch_encode_plus(
            dev_outputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4
        )

        feature = {
            "input_ids": dev_tokenized_input["input_ids"],
            "attention_mask": dev_tokenized_input["attention_mask"],
            "labels": dev_tokenized_output["input_ids"],
            "train_input_ids": train_tokenized_input["input_ids"],
            "train_attention_mask": train_tokenized_input["attention_mask"],
            "train_labels": train_tokenized_output["input_ids"],
            "print_out": {"guid": [qa_data["guid"]]},
            "evaluate": self.evaluate
        }

        if self.evaluate:
            feature["print_out"].update({
                "question": dev_inputs,
                "answer": dev_outputs,
                # "prefix": prefix,
            })

        return feature


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
