import random
import uuid
import torch

from pprint import pprint
from learn2learn.data import MetaDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from .utils.py_io import read_jsonl

dataset_config = {
    "clutrr1": {
        "labels": ["yes", "no", "maybe"],
        "label_to_id": {"yes": 0, "no": 1, "maybe": 2},
        "id_to_label": {0: "yes", 1: "no", 2: "maybe"},
        "num_labels": 3
    },
    "clutrr": {
        "labels": [],
        "label_to_id": {},
        "id_to_label": {},
        "num_labels": 0
    },
    "proofwriter_owa": {
        "labels": ["true", "false", "unknown"],
        "label_to_id": {"true": 0, "false": 1, "unknown": 2},
        "id_to_label": {0: "true", 1: "false", 2: "unknown"},
        "num_labels": 3
    },
    "proofwriter": {
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


def kg_as_autoregressive(triples, rules, baseline=False):
    if baseline:
        facts = [kg['text'] for kg in triples.values()]
        facts += [kg['text'] for kg in rules.values()]
    else:
        facts = [f"triple_{i}: {kg['text']}" for i,
                 kg in enumerate(triples.values())]
        facts.extend([f"rule_{i}: {kg['text']}" for i,
                      kg in enumerate(rules.values())])
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

        true_pairs = []
        false_pairs = []
        unknown_paris = []
        for qa_item in questions.values():
            question = qa_item["question"].replace(".", "")
            triple_enum = [f"triple_{i}" for i in range(len(triples))]
            rule_enum = [f"rule_{i}" for i in range(len(rules))]
            prefix = f"Based on {' '.join(triple_enum)} {' '.join(rule_enum)},"
            question = f"{prefix} can we say {question}?"
            answer = str(qa_item["answer"]).lower()
            if answer == "true":
                true_pairs.append((question, answer))
            elif answer == "false":
                false_pairs.append((question, answer))
            else:
                unknown_paris.append((question, answer))

        qa_pairs = []
        qa_pairs.append(random.choice(true_pairs))
        qa_pairs.append(random.choice(false_pairs))

        if len(unknown_paris) > 0:
            qa_pairs += random.choices(
                unknown_paris,
                k=len(unknown_paris) // 2
            )

        if args.input_format == "mlm":
            facts = kg_as_span_reconstruction(triples, rules)
        else:
            facts = []
            for item in qa_pairs:
                question = item[0].replace(f"{prefix} ", "")
                question = question.replace("?", "")
                parsed_facts = kg_as_autoregressive(
                    triples, rules, args.baseline)

                if args.inner_mode == "closed":
                    prefix = f"To determine if {question}, we need to know"
                    facts.append([f"{prefix} {fact}" for fact in parsed_facts])
                else:
                    facts.append(parsed_facts)

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

        for data in total_qa_data[:2]:
            pprint(data)

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
        proofs = instance["proofs"]
        story = instance["facts"]
        guid = instance["guid"]

        qa_pairs = []
        for qa_item in questions:
            question = qa_item[0].replace("person", "")
            output = f"{qa_item[1]} {qa_item[2]}"
            answer = qa_item[2]
            fact_enum = [f"fact_{i}" for i in range(len(story))]
            prefix = f"Based on {' '.join(fact_enum)}"
            qa_pairs.append((f"{prefix}, {question}", output, answer))

        facts = []
        for item in qa_pairs:
            question = item[0]
            question = question.replace("?", "")
            if args.inner_mode == "closed":
                prefix = f"To determine {question}, we need to know"
                facts.append(
                    [f"{prefix} fact_{i}: {fact}" for i, fact in enumerate(story)])
            elif args.baseline:
                facts.append([fact for fact in story])
            else:
                fact_in = []
                for i, fact_pair in enumerate(story):
                    fact = f"{fact_pair[0]} {fact_pair[1]}"
                    fact_in.append(f"fact_{i}: {fact}")
                facts.append(fact_in)

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
            "clutrr1": ClutrrDataReader,
            "clutrr": ClutrrDataReader
        }
        self.reader = reader_classes[args.dataset_type]
        self.is_training = is_training
        self.logger = logger
        self.tokenizer = tokenizer
        self.dataloader = None
        self.load = False

        self.data = self.read_data_from_file()
        if not self.is_training and args.max_data > 0:
            self.data = random.choices(self.args.max_data)

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
        self.task = args.dataset_type

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
            if args.baseline:
                collate_fn = self.text2text_base_collator
            else:
                collate_fn = self.text2text_collator
        elif args.input_format == "lm":
            if args.baseline:
                collate_fn = self.causal_lm_base_collator
            else:
                collate_fn = self.causal_lm_collator

        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers
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

        max_length = ids1.size(
            1) + ids2.size(1) if max_length is None else max_length
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

        facts_batch = ["\n".join([f"{fact[0]} {fact[1]}" for fact in data['facts']])
                       for data in batch]
        questions = [f"{data['qa_pairs'][0][0]}" for data in batch]
        if len(batch[0]['qa_pairs'][0]) > 2:
            answers = [data['qa_pairs'][0][2] for data in batch]
        else:
            answers = [data['qa_pairs'][0][1] for data in batch]

        max_length = 0
        inputs = []
        for facts, question, answer in zip(facts_batch, questions, answers):
            fact_prefix = f"{bos}{facts}"
            input_txt = f"{fact_prefix}\n{question}"
            if self.args.no_facts:
                input_txt = question
            output_txt = f"{answer}{eos}"
            inputs.append((input_txt, output_txt))

            ids1 = self.tokenizer(
                input_txt,
                return_tensors="pt")["input_ids"]
            ids2 = self.tokenizer(
                output_txt,
                return_tensors="pt")["input_ids"]
            max_length = max(max_length, ids1.size(1) + ids2.size(1))

        input_ids_batch, attention_mask_batch, token_type_ids_batch = [], [], []
        for (txt_in, txt_out) in inputs:
            input_ids, attention_mask, token_type_ids = self._tensorize(
                txt_in, txt_out, max_length)

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            token_type_ids_batch.append(token_type_ids)

        if self.args.classifier:
            labels = torch.LongTensor([self.label_to_id[x] for x in answers])
            labels = torch.nn.functional.one_hot(
                labels,
                num_classes=self.num_classes
            ).to(torch.float)
        else:
            labels = torch.cat(input_ids_batch, dim=0)

        print_inputs = [f"{txt_in}" for (txt_in, _) in inputs]
        if len(batch[0]['qa_pairs'][0]) > 2:
            print_outputs = [data['qa_pairs'][0][2] for data in batch]
        else:
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
        batch_features = []
        for qa_data in batch:
            max_length = 0

            train_input_ids_batch = []
            train_attention_mask_batch = []
            train_token_type_ids_batch = []
            train_samples = []
            for i, fact in enumerate(qa_data['facts']):
                fact_pair = fact.split(':')
                train_input_txt = f"{fact_pair[0].strip()}: "
                train_output_txt = f"{fact_pair[1].strip()}"
                ids1 = self.tokenizer(
                    train_input_txt, return_tensors="pt")["input_ids"]
                ids2 = self.tokenizer(
                    train_output_txt, return_tensors="pt")["input_ids"]
                max_length = max(max_length, ids1.size(1) + ids2.size(1))
                train_samples.append((train_input_txt, train_output_txt))

            for sample in train_samples:
                train_input_txt = sample[0]
                train_output_txt = sample[1]
                input_ids, attention_mask, token_type_ids = self._tensorize(
                    train_input_txt, train_output_txt, max_length)
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
                if len(qa_pair) > 2:
                    dev_output_txt = f"{qa_pair[2]}{eos}"
                else:
                    dev_output_txt = f"{qa_pair[1]}{eos}"
                dev_samples.append(
                    (dev_input_txt, dev_output_txt.replace(eos, '')))
                input_ids, attention_mask, token_type_ids = self._tensorize(
                    dev_input_txt, dev_output_txt)
                dev_input_ids_batch.append(input_ids)
                dev_attention_mask_batch.append(attention_mask)
                dev_token_type_ids_batch.append(token_type_ids)

                if self.args.classifier:
                    labels = torch.nn.functional.one_hot(
                        torch.tensor([self.label_to_id[str(qa_pair[1])]]),
                        num_classes=self.num_classes
                    ).to(torch.float)
                    labels_batch.append(labels)
                else:
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
                "evaluate": self.evaluate
            }

            if self.args.align:
                feature["input_ids"] = feature["train_input_ids"]
                feature["attention_mask"] = feature["train_attention_mask"]
                feature["token_type_ids"] = feature["train_token_type_ids"]

            if self.evaluate:
                train_inputs = [
                    fact[0] for fact in train_samples]
                train_outputs = [
                    fact[1] for fact in train_samples]
                dev_inputs_eval = [sample[0] for sample in dev_samples]
                dev_outputs = [sample[1] for sample in dev_samples]

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

                if self.args.align:
                    feature["print_out"]["question"] = train_inputs
                    feature["print_out"]["answer"] = train_outputs
            batch_features.append(feature)
        return batch_features

    def text2text_base_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        facts_batch = ["\n".join([fact[1]
                                 for fact in data['facts']]) for data in batch]
        questions = [f"{data['qa_pairs'][0][0]}" for data in batch]
        answers = [data['qa_pairs'][0][1] for data in batch]
        inputs = [f"{facts}\n{question}" for facts,
                  question in zip(facts_batch, questions)]

        max_length = 0
        for input in inputs:
            max_length = max(max_length, len(self.tokenizer(input)))

        tokenized_input = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        )

        tokenized_output = self.tokenizer.batch_encode_plus(
            answers,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4
        )

        print_inputs = inputs
        print_outputs = [data['qa_pairs'][0][1] for data in batch]
        print_out = {
            "guid": [data['guid'] for data in batch],
            "prefix": [self.args.dataset for data in batch],
            "question": print_inputs,
            "answer": print_outputs,
        }

        feature = {
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
            "labels": tokenized_output["input_ids"],
            "print_out": print_out,
            "evaluate": self.evaluate
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
