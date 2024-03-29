import torch
import random

from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from .utils.py_io import read_jsonl


def kg_span_reconstruction(text):
    masked = text
    for i, token in enumerate(text.split(" ")):
        masked.replace(token, f"<extra_id_{i}>")
    return masked

class DataReader:
    """Custom dataset loader for QA problems."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        NotImplemented

    @classmethod
    def jsonl_file_reader(cls, path, config):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param evaluation: indicator as to where this is an evaluation file (based on name)
        :param config: the configuration
        """
        total_data = read_jsonl(path)
        all_facts = [fact for data in total_data for fact in data["facts"]]
        all_facts = [fact for data in total_data for fact in data["facts"]]

        total_qa_data = []
        for instance in total_data:
            qa_data = cls._read(instance, config)
            if config.random_facts:
                num_facts = len(qa_data[0]["facts"])
                qa_data[0]["facts"] = random.choices(all_facts, k=num_facts)
                num_facts = len(qa_data[0]["facts"])
                qa_data[0]["facts"] = random.choices(all_facts, k=num_facts)
            total_qa_data += qa_data

        pprint(total_qa_data[0])
        pprint(total_qa_data[0])

        return total_qa_data


class ProofWriterDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        answer_map = {"true": "yes", "false": "no", "unknown": "none"}

        guid = instance["guid"]
        question = instance["question"].replace(".", "")
        answer = answer_map[instance["answer"]]

        if "all_facts" in instance:
            context = [k.replace(".", "") for k in instance["all_facts"]]
            criticals = [k.replace(".", "") for k in instance["facts"]]
            distractors = list(set(context) - set(criticals))
            if args.load_order == "pre":
                context = criticals + distractors
            elif args.load_order == "post":
                context = distractors + criticals
            elif args.load_order == "in":
                random.shuffle(context)
        else:
            context = [k.replace(".", "") for k in instance["facts"]]

        if args.baseline:
            qa_pairs = [[question, answer]]
        else:
            fact_enum = [f"fact_{i}" for i in range(len(context))]
            prefix = f"Based on {' '.join(fact_enum)}"
            qa_pairs = [[f"{prefix}, can we conclude {question}?", answer]]

        if args.multi_task:
            for item in qa_pairs:
                item[1] = f"{item[1]} because {','.join(context)}"

        if not args.baseline:
            context = [f"fact_{i}: {fact}" for i, fact in enumerate(context)]

        return [{"guid": guid, "qa_pairs": [item], "facts": context} for item in qa_pairs]


class ClutrrDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        question = instance["questions"]
        facts = instance["facts"]
        answer = instance["answer"]

        if args.baseline:
            qa_pairs = [[question, answer]]
        else:
            fact_enum = [f"fact_{i}" for i in range(len(facts))]
            prefix = f"Based on {' '.join(fact_enum)}"
            qa_pairs = [[f"{prefix}, {question}", answer]]

        if args.multi_task:
            for item in qa_pairs:
                item[1] = f"{item[1]} because {','.join(facts)}"
        if not args.baseline:
            facts = [f"fact_{i}: {fact}" for i, fact in enumerate(facts)]

        return [{"guid": guid, "qa_pairs": [item], "facts": facts} for item in qa_pairs]


class FolioDataReader(DataReader):
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        answer_map = {
            "true": "yes",
            "false": "no",
            "unknown": "unknown",
            "uncertain": "unknown",
        }
        guid = instance["guid"]
        question = instance["question"]
        context = instance["facts"]
        answer = answer_map[instance["answer"]]

        if args.baseline:
            qa_pairs = [[question, answer]]
        else:
            fact_enum = [f"fact_{i}" for i in range(len(context))]
            prefix = f"Based on {' '.join(fact_enum)}"
            qa_pairs = [[f"{prefix}, can we conclude {question}?", answer]]

        if args.multi_task:
            for item in qa_pairs:
                item[1] = f"{item[1]} because {','.join(context)}"

        if not args.baseline:
            context = [f"fact_{i}: {fact}" for i, fact in enumerate(context)]

        return [{"guid": guid, "qa_pairs": [item], "facts": context} for item in qa_pairs]


class EntailmentTreeDataReader:
    """Custom dataset loader for QA problems with associated knowledge facts."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        question = instance["hypothesis"]
        context = instance["facts"]
        answer = instance["answer"]

        fact_enum = [f"fact_{i}" for i in range(len(context))]
        prefix = f"Based on {' '.join(fact_enum)}"
        if args.baseline:
            qa_pairs = [[question, answer]]
        else:
            qa_pairs = [[f"{prefix}, can we conclude {question}?", answer]]

        if args.multi_task:
            for item in qa_pairs:
                item[1] = f"{item[1]} because {','.join(context)}"

        if not args.baseline:
            facts = [f"fact_{i}: {fact}" for i, fact in enumerate(context)]
        else:
            facts = context

        return [{"guid": guid, "qa_pairs": [item], "facts": facts} for item in qa_pairs]

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

        for data in total_qa_data[:2]:
            pprint(data)

        return total_qa_data


class MetaKnowledgeDataset(object):
    def __init__(self, logger, args, tokenizer, data_path, data_type, is_training):
        self.data_path = data_path
        self.data_type = data_type
        self.task_name = args.dataset
        self.args = args

        reader_classes = {
            "proofwriter": ProofWriterDataReader,
            "clutrr": ClutrrDataReader,
            "folio": FolioDataReader,
            "entailment_tree": EntailmentTreeDataReader,
        }
        self.reader = reader_classes[args.dataset_type]
        self.is_training = is_training
        self.logger = logger
        self.tokenizer = tokenizer
        self.dataloader = None
        self.load = False

        self.data = self.read_data_from_file()
        if not self.is_training and args.max_data > 0:
            self.data = random.choices(self.data, k=args.max_data)

        if self.is_training and args.do_eval:
            self.data = self.data[:1]

    def __len__(self):
        return len(self.data)

    def read_data_from_file(self):
        file_path = f"{self.data_path}/{self.task_name}/{self.data_type}.jsonl"
        file_data = self.reader.jsonl_file_reader(file_path, self.args)
        return file_data

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers) + len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataloader(self):
        meta_dataloader = MetaDataLoader(
            self.args, self.data, self.tokenizer, self.is_training
        )
        self.dataloader = meta_dataloader.dataloader
        return self.dataloader

class MetaDataLoader:
    def __init__(self, args, dataset, tokenizer, is_training):
        self.args = args
        self.task = args.dataset_type

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.evaluate = not is_training

        if is_training:
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        if args.model_type == "t5":
            if args.baseline:
                collate_fn = self.text2text_baseline_collator
            else:
                collate_fn = self.text2text_collator
        else:
            if args.baseline:
                collate_fn = self.causal_lm_base_collator
            else:
                collate_fn = self.causal_lm_collator
        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    def _tensorize(self, input_txt, output_txt, max_length=None):
        """Converts a list of strings into a tensor of ids

        :param input_txt: the input text
        :param output_txt: the output text
        :param max_length: the maximum length of the sequence
        :return: the tensorized input and output
        """
        pad = self.tokenizer.eos_token
        pad_id = self.tokenizer.encode(pad)

        ids1 = self.tokenizer(input_txt, return_tensors="pt")["input_ids"]
        ids2 = self.tokenizer(output_txt, return_tensors="pt")["input_ids"]

        max_length = ids1.size(1) + ids2.size(1) if max_length is None else max_length
        max_length = ids1.size(1) + ids2.size(1) if max_length is None else max_length
        n_mask = max_length - ids1.size(1) - ids2.size(1)
        assert n_mask >= 0, (max_length, ids1.size(1), ids2.size(1))
        padding = torch.LongTensor(pad_id * n_mask).unsqueeze(0)

        input_ids = torch.cat((ids1, ids2, padding), dim=1)
        attention_mask = torch.LongTensor(
            [1] * (ids1.size(1) + ids2.size(1)) + [0] * n_mask
        ).unsqueeze(0)
        token_type_ids = torch.LongTensor(
            [0] * ids1.size(1) + [1] * ids2.size(1) + [0] * n_mask
        ).unsqueeze(0)

        assert (
            input_ids.size(1)
            == attention_mask.size(1)
            == token_type_ids.size(1)
            == max_length
        )
        assert (
            input_ids.size(1)
            == attention_mask.size(1)
            == token_type_ids.size(1)
            == max_length
        )

        return input_ids, attention_mask, token_type_ids

    def causal_lm_base_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        eos = self.tokenizer.eos_token
        bos = self.tokenizer.bos_token

        facts_batch = ["\n".join([fact for fact in data["facts"]]) for data in batch]
        facts_batch = ["\n".join([fact for fact in data["facts"]]) for data in batch]
        questions = [f"{data['qa_pairs'][0][0]}" for data in batch]
        answers = [data["qa_pairs"][0][1] for data in batch]
        answers = [data["qa_pairs"][0][1] for data in batch]

        max_length = 0
        inputs = []
        for facts, question, answer in zip(facts_batch, questions, answers):
            fact_prefix = f"{bos}{facts}"
            input_txt = f"{fact_prefix}\n{question}"


            if self.args.no_facts:
                input_txt = question
            elif self.args.no_question and not self.evaluate:
                input_txt = fact_prefix
            elif self.args.no_question and self.evaluate:
                input_txt = question

            if self.args.no_question and not self.evaluate:
                output_txt = f"{fact_prefix}"
            else:
                output_txt = f"{answer}{eos}"

            inputs.append((input_txt, output_txt))

            ids1 = self.tokenizer(input_txt, return_tensors="pt")["input_ids"]
            ids2 = self.tokenizer(output_txt, return_tensors="pt")["input_ids"]
            max_length = max(max_length, ids1.size(1) + ids2.size(1))

        input_ids_batch, attention_mask_batch, token_type_ids_batch = [], [], []
        for txt_in, txt_out in inputs:
            input_ids, attention_mask, token_type_ids = self._tensorize(
                txt_in, txt_out, max_length)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            token_type_ids_batch.append(token_type_ids)

        print_inputs = [txt_in for (txt_in, _) in inputs]
        if len(batch[0]["qa_pairs"][0]) > 2:
            print_outputs = [data["qa_pairs"][0][2] for data in batch]
        if len(batch[0]["qa_pairs"][0]) > 2:
            print_outputs = [data["qa_pairs"][0][2] for data in batch]
        else:
            print_outputs = [data["qa_pairs"][0][1] for data in batch]
            print_outputs = [data["qa_pairs"][0][1] for data in batch]
        print_out = {
            "guid": [data["guid"] for data in batch],
            "guid": [data["guid"] for data in batch],
            "prefix": [self.args.dataset for data in batch],
            "question": print_inputs,
            "answer": print_outputs,
        }

        feature = {
            "input_ids": torch.cat(input_ids_batch, dim=0),
            "attention_mask": torch.cat(attention_mask_batch, dim=0),
            "token_type_ids": torch.cat(token_type_ids_batch, dim=0),
            "labels": torch.cat(input_ids_batch, dim=0),
            "print_out": print_out,
            "evaluate": self.evaluate,
        }

        return feature

    def causal_lm_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        eos = self.tokenizer.eos_token
        batch_features = []
        for qa_data in batch:
            max_length = 0

            train_input_ids_batch = []
            train_attention_mask_batch = []
            train_token_type_ids_batch = []
            train_samples = []
            for fact in qa_data["facts"]:
                fact_pair = fact.split(":")
            for fact in qa_data["facts"]:
                fact_pair = fact.split(":")
                train_input_txt = f"{fact_pair[0].strip()}: "
                train_output_txt = f"{fact_pair[1].strip()}"
                ids1 = self.tokenizer(train_input_txt, return_tensors="pt")["input_ids"]
                ids2 = self.tokenizer(train_output_txt, return_tensors="pt")[
                    "input_ids"
                ]
                ids1 = self.tokenizer(train_input_txt, return_tensors="pt")["input_ids"]
                ids2 = self.tokenizer(train_output_txt, return_tensors="pt")[
                    "input_ids"
                ]
                max_length = max(max_length, ids1.size(1) + ids2.size(1))
                train_samples.append((train_input_txt, train_output_txt))

            for sample in train_samples:
                train_input_txt = sample[0]
                train_output_txt = sample[1]
                input_ids, attention_mask, token_type_ids = self._tensorize(
                    train_input_txt, train_output_txt, max_length
                )
                train_input_ids_batch.append(input_ids)
                train_attention_mask_batch.append(attention_mask)
                train_token_type_ids_batch.append(token_type_ids)

            dev_input_ids_batch = []
            dev_attention_mask_batch = []
            dev_token_type_ids_batch = []
            labels_batch = []
            dev_samples = []
            for qa_pair in qa_data["qa_pairs"]:
                dev_input_txt = qa_pair[0]
                dev_output_txt = f"{qa_pair[1]}{eos}"
                dev_samples.append((dev_input_txt, dev_output_txt.replace(eos, "")))
                dev_samples.append((dev_input_txt, dev_output_txt.replace(eos, "")))
                input_ids, attention_mask, token_type_ids = self._tensorize(
                    dev_input_txt, dev_output_txt
                )
                dev_input_ids_batch.append(input_ids)
                dev_attention_mask_batch.append(attention_mask)
                dev_token_type_ids_batch.append(token_type_ids)

                labels_batch.append(input_ids)

            feature = {
                "input_ids": torch.cat(dev_input_ids_batch, dim=0),
                "attention_mask": torch.cat(dev_attention_mask_batch, dim=0),
                "token_type_ids": torch.cat(dev_token_type_ids_batch, dim=0),
                "labels": torch.cat(labels_batch, dim=0),
                "train_input_ids": torch.cat(train_input_ids_batch, dim=0),
                "train_attention_mask": torch.cat(train_attention_mask_batch, dim=0),
                "train_token_type_ids": torch.cat(train_token_type_ids_batch, dim=0),
                "print_out": {"guid": [qa_data["guid"]]},
                "evaluate": self.evaluate,
            }

            if self.evaluate:
                train_inputs = [fact[0] for fact in train_samples]
                train_outputs = [fact[1] for fact in train_samples]
                dev_inputs_eval = [sample[0] for sample in dev_samples]
                dev_outputs = [sample[1] for sample in dev_samples]

                feature["print_out"].update(
                    {
                        "question": dev_inputs_eval,
                        "answer": dev_outputs,
                        "prefix": [self.args.dataset],
                    }
                )

                feature["inner_print_out"] = {
                    "prompt": train_inputs,
                    "fact": train_outputs,
                    "guid": [qa_data["guid"]],
                }
            batch_features.append(feature)
        return batch_features
    def text2text_baseline_collator(
        self,
        batch,
    ):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        facts_batch = ["\n".join([fact for fact in data["facts"]]) for data in batch]
        questions = [f"{data['qa_pairs'][0][0]}" for data in batch]
        answers = [data["qa_pairs"][0][1] for data in batch]

        inputs, outputs = [], []
        for facts, question, answer in zip(facts_batch, questions, answers):
            inputs.append(f"{facts}\n{question}")
            outputs.append(answer)

        input_max_length = max([len(self.tokenizer.encode(i)) for i in inputs])
        output_max_length = max([len(self.tokenizer.encode(o)) for o in outputs])

        tokenized_inputs = self.tokenizer(
            inputs,
            max_length=input_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokenized_outputs = self.tokenizer(
            outputs,
            max_length=output_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if len(batch[0]["qa_pairs"][0]) > 2:
            print_outputs = [data["qa_pairs"][0][2] for data in batch]
        else:
            print_outputs = [data["qa_pairs"][0][1] for data in batch]

        print_out = {
            "guid": [data["guid"] for data in batch],
            "prefix": [self.args.dataset for _ in batch],
            "question": inputs,
            "answer": print_outputs,
        }

        feature = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_outputs["input_ids"],
            "print_out": print_out,
            "evaluate": self.evaluate,
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
        batch_features = []
        for qa_data in batch:
            train_input_txt_batch = []
            train_output_txt_batch = []
            for fact in qa_data["facts"]:
                fact_pair = fact.split(":")
                train_output_txt = fact_pair[1].strip()
                masked = kg_span_reconstruction(train_output_txt)
                train_input_txt = f"{fact_pair[0].strip()}: {masked}"
                train_input_txt_batch.append(train_input_txt)
                train_output_txt_batch.append(train_output_txt)

            input_max_length = [
                len(self.tokenizer.encode(e)) for e in train_input_txt_batch
            ]
            output_max_length = [
                len(self.tokenizer.encode(e)) for e in train_output_txt_batch
            ]

            train_tokenized_input = self.tokenizer.batch_encode_plus(
                train_input_txt_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max(input_max_length),
            )
            train_tokenized_output = self.tokenizer.batch_encode_plus(
                train_output_txt_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max(output_max_length),
            )

            dev_input_txt_batch = []
            dev_output_txt_batch = []
            labels_batch = []
            dev_samples = []
            for qa_pair in qa_data["qa_pairs"]:
                dev_input_txt_batch.append(qa_pair[0])
                dev_output_txt_batch.append(qa_pair[1])
                labels_batch.append(qa_pair[1])
                dev_samples.append((qa_pair[0], qa_pair[1]))

            input_max_length = [
                len(self.tokenizer.encode(e)) for e in dev_input_txt_batch
            ]
            output_max_length = [
                len(self.tokenizer.encode(e)) for e in dev_output_txt_batch
            ]

            dev_tokenized_input = self.tokenizer.batch_encode_plus(
                dev_input_txt_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max(input_max_length),
            )
            dev_tokenized_output = self.tokenizer.batch_encode_plus(
                dev_output_txt_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max(output_max_length),
            )

            feature = {
                "input_ids": dev_tokenized_input["input_ids"],
                "attention_mask": dev_tokenized_input["attention_mask"],
                "labels": dev_tokenized_output["input_ids"],
                "train_input_ids": train_tokenized_input["input_ids"],
                "train_attention_mask": train_tokenized_input["attention_mask"],
                "train_labels": train_tokenized_output["input_ids"],
                "print_out": {"guid": [qa_data["guid"]]},
                "evaluate": self.evaluate,
            }

            if self.evaluate:
                dev_inputs_eval = [sample[0] for sample in dev_samples]
                dev_outputs = [sample[1] for sample in dev_samples]
                feature["print_out"].update(
                    {
                        "question": dev_inputs_eval,
                        "answer": dev_outputs,
                        "guid": [qa_data["guid"]],
                    }
                )
            batch_features.append(feature)
        return batch_features