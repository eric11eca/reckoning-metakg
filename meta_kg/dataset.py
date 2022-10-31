import os
import re
import random
import uuid
import string
import numpy as np

from collections import Counter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .utils.py_io import read_jsonl

dataset_config = {
    "proofwriter_owa_natlang": 3,
    "proofwriter_cwa_natlang": 2,
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


class MetaQADataReader():
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

        qa_pairs = []
        unknown_paris = []
        for qa_item in questions.values():
            question = qa_item["question"].replace(".", "")
            question = f"Is it true, false, or unknown that {question}?"
            question = f"Is it true or false that {question}?"
            answer = str(qa_item["answer"]).lower()
            if answer != "unknown":
                qa_pairs.append((question, answer))
            else:
                unknown_paris.append((question, answer))

        qa_pairs += random.choices(unknown_paris, k=len(unknown_paris) // 2)

        if args.input_format == "mlm":
            facts = kg_as_span_reconstruction(triples, rules)
        else:
            facts = []
            for item in qa_pairs:
                # question = item[0].replace("Is it true, false, or unknown that ", "")
                question = item[0].replace("Is it true or false that ", "")
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
    data_container = MetaQADataReader.jsonl_file_reader(
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
        self.data = self.read_data_from_file()

        self.is_training = is_training
        self.logger = logger
        self.tokenizer = tokenizer
        self.dataloader = None
        self.load = False

        print(self.data[:2])

    def __len__(self):
        return len(self.data)

    def read_data_from_file(self):
        file_path = f"{self.data_path}/{self.task_name}/{self.data_type}.jsonl"
        file_data = MetaQADataReader.jsonl_file_reader(file_path, self.args)
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
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.evaluate = not is_training

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

        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

    def _tokenize_collate_fn(self, batch):
        return self.tokenizer.batch_encode_plus(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=32
        )

    def causal_lm_base_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """

        eos = self.tokenizer.eos_token
        bos = self.tokenizer.bos_token
        gen = "<gen>"

        facts_batch = [[fact[1] for fact in data['facts']] for data in batch]
        questions = [f"{data['qa_pairs'][0][0]} {gen} {data['qa_pairs'][0][1]}{eos}" for data in batch]

        inputs = []
        for facts, question in zip(facts_batch, questions):
            fact_prefix = f"{bos}"
            for fact in facts:
                fact_prefix += f"{fact} "
            inputs.append(f"{fact_prefix} {question}")
        
        dev_tokenized_input = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=16
        )



    def causal_lm_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """

        eos = self.tokenizer.eos_token
        bos = self.tokenizer.bos_token
        gen = "<gen>"
        enc = "<enc>"

        qa_data = batch[0]
        train_inputs = []
        for i, fact in enumerate(qa_data["facts"]):
            # train_inputs.append(f"{fact[0]} fact_{i} {enc} {fact[1]}")
            train_inputs.append(f"{bos}fact_{i} {enc} {fact[1]}{eos}")

        train_outputs = [
            fact[1] for fact in qa_data["facts"]
        ]

        train_tokenized = self._tokenize_collate_fn(train_inputs)

        dev_inputs = [
            f"{bos}{qa_pair[0]} {gen} {qa_pair[1]}{eos}".replace("unknown", "none") for qa_pair in qa_data["qa_pairs"]]

        dev_inputs_eval = [
            f"{bos}{qa_pair[0]} {gen}".replace("unknown", "none") for qa_pair in qa_data["qa_pairs"]]
        dev_outputs = [str(qa_pair[1]).replace("unknown", "none") for qa_pair in qa_data["qa_pairs"]]

        dev_tokenized_input = self.tokenizer.batch_encode_plus(
            dev_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=16
        )

        feature = {
            "input_ids": dev_tokenized_input["input_ids"],
            "attention_mask": dev_tokenized_input["attention_mask"],
            "labels": dev_tokenized_input["input_ids"],
            "train_input_ids": train_tokenized["input_ids"],
            "train_attention_mask": train_tokenized["attention_mask"],
            "print_out": {"guid": [qa_data["guid"]]},
            "evaluate": self.evaluate
        }

        if self.evaluate:
            feature["print_out"].update({
                "question": dev_inputs_eval,
                "answer": dev_outputs,
                # "prefix": prefix,
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
