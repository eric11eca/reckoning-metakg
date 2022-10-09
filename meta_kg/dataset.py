import os
import re
import json
import uuid
import string
import numpy as np

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .utils import MyMetaLearningDataset, MyMetaLearningDataLoader
from .reader import JsonlReader


@dataclass
class MetaQAInput:
    """Base class for input examples

    :param text: the input text
    :param guid: the instance identifier (optional)
    """
    guid: str = field(default='')
    question: str = field(default='')
    answer: str = field(default='')
    facts: List[float] = field(default_factory=list)
    evaluation: bool = False
    prefix: str = ""

    def __str__(self):
        return "<InputExample> label: {}, question: {}, answer: {}, evaluation={}, prefix={}".format(
            self.label, self.question,
            self.answer, str(self.evaluation), self.prefix,
        )

    @classmethod
    def from_json(cls, json_instance):
        question = json_instance["question"] if "question" in json_instance else None
        answer = json_instance["answer"] if "answer" in json_instance else None
        facts = json_instance["facts"] if "facts" in json_instance else None
        guid = json_instance["guid"] if "guid" in json_instance else None
        return cls(guid, question, answer, facts)


class MetaQADataReader(JsonlReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"] if "guid" in instance else str(uuid.uuid1())
        question = instance["question"]
        answer = instance["answer"]
        facts = instance["facts"]

        return MetaQAInput(
            guid=guid,
            question=question,
            answer=answer,
            facts=facts
        )

    @classmethod
    def json_file_reader(cls, path, evaluation):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param evaluation: indicator as to where this is an evaluation file (based on name)
        :param config: the configuration
        """
        total_data = []

        with open(path) as my_json:
            for k, line in enumerate(my_json):
                line = line.strip()
                json_line = json.loads(line)
                if not json_line["output"] or json_line["output"] == "none":
                    continue

                line_instance = cls._read(json_line)
                line_instance.evaluation = evaluation

                total_data.append(line_instance)

                # if k <= 3:
                #     debug_line = f"""===========\nraw line \n {json.dumps(json_line,indent=4)} \n processed line \n {line_instance}\n============\n"""
                #     logger.info(debug_line)

        return total_data


def meta_qa_data_loader(
    data_path: str,
    split: str,
    config=None,
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

    data_container = MetaQADataReader.from_file(
        target_data,
        config=config,
        evaluate=evaluate
    )

    return data_container.data


class MetaKnowledgeDataset(object):
    def __init__(self, logger, args, data_path, data_type, is_training):
        self.data_path = data_path
        self.data_type = data_type
        self.data = []
        self.meta_qa_data = meta_qa_data_loader(
            data_path,
            split="train",
            config=args,
            evaluate=False
        )

        for qa_data in self.meta_qa_data:
            train_examples = qa_data["facts"]
            dev_examples = qa_data["qa_pairs"]

            self.data.append({
                "guid": qa_data["guid"],
                "train_examples": train_examples,
                "dev_examples": dev_examples
            })

        self.is_training = is_training
        self.logger = logger
        self.args = args
        self.metric = "acc"
        self.tokenizer = None
        self.dataset = None
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
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        split_identifier = self.args.custom_tasks_splits.split("/")[-1]
        if split_identifier.endswith(".json"):
            split_identifier = split_identifier[:-5]

        preprocessed_path = os.path.join(
            self.data_path,
            self.data_type +
            "-meta-{}-{}.json".format(split_identifier, postfix)
        )

        if self.load and os.path.exists(preprocessed_path):
            self.logger.info(
                "Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                train_input_ids, train_attention_mask, \
                    train_decoder_input_ids, train_decoder_attention_mask, \
                    train_metadata_task, train_metadata_questions, \
                    dev_input_ids, dev_attention_mask, \
                    dev_decoder_input_ids, dev_decoder_attention_mask, \
                    dev_metadata_task, dev_metadata_questions = json.load(f)

        else:
            self.logger.info(
                "Start tokenizing ... {} instances".format(len(self.data)))

            train_inputs = []
            train_outputs = []
            dev_inputs = []
            dev_outputs = []

            train_metadata_task, train_metadata_questions = [], []
            train_st, train_ed = 0, 0
            dev_metadata_task, dev_metadata_questions = [], []
            dev_st, dev_ed = 0, 0

            for qa_data in self.data:
                guid = qa_data["guid"]
                for dp in qa_data["train_examples"]:
                    train_inputs.append({dp[0]})
                    train_outputs.append([" " + item for item in dp[1]])

                train_st = train_ed
                train_ed = train_ed + len(qa_data["train_examples"])
                train_metadata_task.append((train_st, train_ed))

                for dp in qa_data["dev_examples"]:
                    dev_inputs.append({dp[0]})
                    dev_outputs.append([" " + item for item in dp[1]])

                dev_st = dev_ed
                dev_ed = dev_ed + len(qa_data["dev_examples"])
                dev_metadata_task.append((dev_st, dev_ed))

            train_outputs, train_metadata_questions = self.flatten(
                train_outputs)
            dev_outputs, dev_metadata_questions = self.flatten(dev_outputs)

            self.logger.info("Printing 3 examples")
            for i in range(3):
                self.logger.info(dev_inputs[i])
                self.logger.info(dev_outputs[i])

            self.logger.info("Tokenizing Train Input ...")
            train_tokenized_input = tokenizer.batch_encode_plus(train_inputs,
                                                                pad_to_max_length=True,
                                                                max_length=self.args.max_input_length)
            self.logger.info("Tokenizing Train Output ...")
            train_tokenized_output = tokenizer.batch_encode_plus(train_outputs,
                                                                 pad_to_max_length=True,
                                                                 max_length=self.args.max_output_length)

            self.logger.info("Tokenizing Dev Input ...")
            dev_tokenized_input = tokenizer.batch_encode_plus(dev_inputs,
                                                              pad_to_max_length=True,
                                                              max_length=self.args.max_input_length)
            self.logger.info("Tokenizing Dev Output ...")
            dev_tokenized_output = tokenizer.batch_encode_plus(dev_outputs,
                                                               pad_to_max_length=True,
                                                               max_length=self.args.max_output_length)

            train_input_ids, train_attention_mask = train_tokenized_input[
                "input_ids"], train_tokenized_input["attention_mask"]
            train_decoder_input_ids, train_decoder_attention_mask = train_tokenized_output[
                "input_ids"], train_tokenized_output["attention_mask"]

            dev_input_ids, dev_attention_mask = dev_tokenized_input[
                "input_ids"], dev_tokenized_input["attention_mask"]
            dev_decoder_input_ids, dev_decoder_attention_mask = dev_tokenized_output[
                "input_ids"], dev_tokenized_output["attention_mask"]

            if self.load:

                with open(preprocessed_path, "w") as f:
                    json.dump([train_input_ids, train_attention_mask,
                               train_decoder_input_ids, train_decoder_attention_mask,
                               train_metadata_task, train_metadata_questions,
                               dev_input_ids, dev_attention_mask,
                               dev_decoder_input_ids, dev_decoder_attention_mask,
                               dev_metadata_task, dev_metadata_questions
                               ], f)

        self.dataset = MyMetaLearningDataset(train_input_ids, train_attention_mask,
                                             train_decoder_input_ids, train_decoder_attention_mask,
                                             train_metadata_task, train_metadata_questions,
                                             dev_input_ids, dev_attention_mask,
                                             dev_decoder_input_ids, dev_decoder_attention_mask,
                                             dev_metadata_task, dev_metadata_questions,
                                             inner_bsz=self.args.inner_bsz,
                                             is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(
            len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyMetaLearningDataLoader(
            self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        # not used
        return 0.0

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
